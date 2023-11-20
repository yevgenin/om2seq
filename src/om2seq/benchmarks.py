from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer
from typing import Optional

import more_itertools
import numpy as np
import pandas as pd
import wandb
from datasets import Dataset
from devtools import debug
from fire import Fire
from tqdm import tqdm

from deepom.aligner import DeepOMAligner
from deepom.localizer import DeepOMLocalizer
from om2seq.cropping import Cropper
from om2seq.embeddings import EmbeddingsDB
from om2seq.env import ENV
from om2seq.inference import InferenceModel
from om2seq.mapping import MappingResult, RefEmb, QryEmb, RefSegment
from om2seq.plotting import CropPlot
from om2seq.train import Metrics
from om2seq.data import TrainingDataset, GenomeDataset
from utils.alignment import AlignmentInfo, AlignedImage
from utils.dataset_tasks import BaseTask, ParallelTask
from utils.pyutils import PydanticClassConfig, PydanticClassInputs

QRY_LEN = 'DNA fragment image length (kbp)'


class BenchmarkMetrics(Metrics):
    time: Optional[float] = None


class Benchmark(BaseTask):
    class Config(PydanticClassConfig):
        min_len: int = TrainingDataset.Config().min_len
        max_len: int = TrainingDataset.Config().max_len
        num_len: int = 16
        ref_limit: Optional[int] = None
        ref_emb_limit: Optional[int] = None
        qry_limit: Optional[int] = None
        length_min_delta: int = 10000
        model_id_wandb_run_name: str = '89fw8ce7'  # 'sjksath7'
        enable_om2seq: bool = True
        enable_deepom: bool = True
        enable_combined: bool = True

    config: Config

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_db = None
        self.aligner = None
        self.localizer = None
        self.training_dataset: TrainingDataset = None
        self.ref_emb_ds = None
        self.inference_model = None
        self.references = GenomeDataset(limit=self.config.ref_limit).references()

    def init_db_task(self):
        self.init_om2seq()

    def init_om2seq(self):
        self.inference_model = InferenceModel(model_id_wandb_run_name=self.config.model_id_wandb_run_name)
        self.ref_db = EmbeddingsDB(limit=self.config.ref_emb_limit,
                                   references=self.references,
                                   model_id=self.config.model_id_wandb_run_name,
                                   inference_model=self.inference_model)
        self.ref_emb_ds = self.ref_db.dataset()

    def init_dataset(self):
        self.training_dataset = TrainingDataset(references=self.references, add_random_shift=False,
                                                add_random_flip=False)

    def benchmark(self):
        with self.wandb_init():
            self.init_dataset()
            if self.config.enable_om2seq or self.config.enable_combined:
                self.init_om2seq()

            if self.config.enable_deepom or self.config.enable_combined:
                self.init_deepom()

            df = self.compute_metrics()
            wandb.log({'metrics': wandb.Table(dataframe=df)})
            print(df)

    def init_deepom(self):
        self.localizer = DeepOMLocalizer(device='cuda')
        self.aligner = DeepOMAligner()

    def generate_crops(self, qry_len: int, limit: int = None):
        return self.training_dataset.generate_crops_dataset(
            aligned_images_dataset=self.training_dataset.training_split['test'],
            qry_len=qry_len,
            limit=limit
        )

    def eval_combined(self, crops):
        return EvalCombined(
            inference_model=self.inference_model,
            ref_emb_ds=self.ref_emb_ds,
            localizer=self.localizer,
            aligner=self.aligner,
            references=self.references,
            qry_limit=self.config.qry_limit,
            crops=crops,
        )

    def eval_om2seq(self, crops):
        return EvalOM2Seq(
            inference_model=self.inference_model,
            ref_emb_ds=self.ref_emb_ds,
            qry_limit=self.config.qry_limit,
            crops=crops,
        )

    def eval_deepom(self, crops):
        return EvalDeepOM(
            localizer=self.localizer,
            aligner=self.aligner,
            references=self.references,
            qry_limit=self.config.qry_limit,
            crops=crops,
        )

    def compute_metrics(self):
        lengths = self._get_lengths()
        debug(lengths)

        crop_sets = ((qry_len, self.generate_crops(qry_len=qry_len)) for qry_len in lengths)
        evaluation_sets = (
            (
                qry_len, [
                    self.eval_om2seq(crops) if self.config.enable_om2seq else None,
                    self.eval_deepom(crops) if self.config.enable_deepom else None,
                    self.eval_combined(crops) if self.config.enable_combined else None,
                ]
            )
            for qry_len, crops in crop_sets
        )
        metrics = (
            dict(qry_len=qry_len, **{
                evaluation.evaluation_name + '_' + key: value
                for evaluation in evaluation_set
                if evaluation is not None
                for key, value in evaluation.compute_metrics()
            })
            for qry_len, evaluation_set in evaluation_sets
        )
        return pd.DataFrame(debug(_) for _ in metrics)

    def _get_lengths(self):
        lengths = np.geomspace(self.config.min_len, self.config.max_len, self.config.num_len).astype(int)
        lengths = np.unique((lengths // self.config.length_min_delta * self.config.length_min_delta).astype(int))
        return lengths


class OMEvaluation(BaseTask):
    INDEX_NAME = EmbeddingsDB.INDEX_NAME

    class Config(PydanticClassConfig):
        qry_limit: Optional[int] = None

    class Inputs(PydanticClassInputs):
        crops: Dataset = None

    config: Config

    @staticmethod
    def accuracy_metrics(correct: list[bool], time: float = None):
        correct = np.asarray(correct)
        return BenchmarkMetrics(
            accuracy=correct.mean(),
            num_correct=correct.sum(),
            num_total=len(correct),
            time=time,
        )

    @staticmethod
    def segment_overlap(pred_start, pred_stop, start, stop):
        overlap = min(pred_stop, stop) - max(pred_start, start)
        return overlap

    def compute_metrics(self):
        start = default_timer()
        correct = self.compute_correctness()
        end = default_timer()
        return self.accuracy_metrics(correct=correct, time=end - start)

    def get_crops(self):
        return more_itertools.take(self.config.qry_limit, self.inputs.crops)

    def compute_correctness(self) -> list[bool]:
        raise NotImplementedError


class EvalOM2Seq(OMEvaluation):
    evaluation_name = 'OM2Seq'

    class Config(OMEvaluation.Config):
        top_k: int = 1

    class Inputs(OMEvaluation.Inputs):
        inference_model: InferenceModel
        ref_emb_ds: Dataset

    config: Config
    inputs: Inputs

    def compute_correctness(self) -> list[bool]:
        return [self.top_result(_).correct for _ in self.mapping_results()]

    def mapping_results(self) -> list[list[MappingResult]]:
        with debug.timer('faiss search'):
            retrieved = self.retrieve(query_embeddings=self.inference(queries=self.get_crops()))

        return [
            [
                self._mapping_result(ref=ref, qry=qry, score=score)
                for qry, ref, score in ref_list
            ]
            for ref_list in retrieved
        ]

    def retrieve(self, query_embeddings: list[QryEmb]):
        query_emb_vectors = np.stack([_.qry_emb for _ in query_embeddings])
        search_results = self.search_batch(query_emb_vectors)
        return [
            [
                (qry, RefEmb(**self.inputs.ref_emb_ds[int(index)]), score)
                for index, score in zip(indices, scores)
            ]
            for qry, indices, scores in zip(query_embeddings, search_results.total_indices, search_results.total_scores)
        ]

    def inference(self, queries: list[Cropper.AlignedCrop]) -> list[QryEmb]:
        return self.inputs.inference_model.inference_batched(items=queries,
                                                             qry_limit=self.config.qry_limit)

    def search_batch(self, queries: np.ndarray, **kwargs):
        return self.inputs.ref_emb_ds.search_batch(index_name=EmbeddingsDB.INDEX_NAME,
                                                   queries=queries,
                                                   k=self.config.top_k, **kwargs)

    def _mapping_result(self, ref: RefEmb, qry: QryEmb, score: float):
        overlap = self.segment_overlap(ref.ref_start, ref.ref_stop, qry.ref_start, qry.ref_stop)
        return MappingResult(
            qry=qry,
            ref=ref,
            correct=(ref.reference_id == qry.reference_id) and (overlap > 0),
            overlap=overlap,
            score=score,
        )

    def top_result(self, mrs: list[MappingResult]):
        #   the first result is the best, as returned by faiss
        return mrs[0]


class EvalDeepOM(OMEvaluation):
    evaluation_name = 'DeepOM'

    class Config(OMEvaluation.Config, ParallelTask.Config):
        nominal_scale: float = ENV.NOMINAL_SCALE

    class Inputs(OMEvaluation.Inputs):
        localizer: DeepOMLocalizer
        aligner: DeepOMAligner
        references: dict[str, np.ndarray]

    def __init__(self, **kwargs):
        self.config = self.Config(**kwargs)
        if hasattr(self, 'Inputs'):
            self.inputs = self.Inputs(**kwargs)

    class DeepOMCrop(Cropper.AlignedCrop, AlignedImage, DeepOMLocalizer.LocalizerOutput):
        pass

    class DeepOMMappingResult(DeepOMAligner.Alignment):
        correct: bool
        overlap: Optional[int] = None
        overlap_ratio: float = None

    config: Config

    def compute_correctness(self) -> list[bool]:
        inputs = self.deepom_preprocess()

        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            results = executor.map(lambda _: self.deepom_mapping_result(_).correct, inputs)
            return list(tqdm(results, total=len(inputs), desc=self.compute_correctness.__name__))

    def deepom_localize(self, crop: dict):
        inference = self.inputs.localizer.inference(Cropper.AlignedCrop(**crop).crop_image, preprocess_image=False,
                                                    extras=True)
        return self.DeepOMCrop(**crop, **dict(inference))

    def deepom_preprocess(self):
        inputs = self.get_crops()
        with ThreadPoolExecutor(max_workers=1) as executor:
            crops = tqdm(
                executor.map(self.deepom_localize, inputs),
                total=len(inputs), desc=self.deepom_preprocess.__name__
            )

        return [_ for _ in crops if len(_.localizations) >= 2]

    def deepom_mapping_result(self, crop: DeepOMCrop):
        alignment = self.top_alignment(crop)
        pred_start, pred_stop = AlignmentInfo(**dict(alignment), references=self.inputs.references).ref_lims
        start, stop = AlignmentInfo(**dict(crop), references=self.inputs.references).ref_lims
        overlap = self.segment_overlap(pred_start, pred_stop, start, stop)

        return self.DeepOMMappingResult(
            **dict(alignment),
            correct=(
                    (alignment.reference_id == crop.reference_id)
                    and (overlap > 0)
            ),
            overlap=overlap,
            overlap_ratio=(overlap / (pred_stop - pred_start)),
        )

    def top_alignment(self, crop: DeepOMCrop):
        return self.inputs.aligner.align(
            query=self.inputs.aligner.Query(
                query_positions=crop.localizations,
                query_scale=self.config.nominal_scale,
            ),
            references=self.inputs.references
        )


class EvalCombined(EvalOM2Seq):
    evaluation_name = 'OM2Seq+DeepOM'

    class Config(EvalOM2Seq.Config):
        top_k: int = 16
        margin: int = 0

    class Inputs(EvalOM2Seq.Inputs, EvalDeepOM.Inputs):
        pass

    config: Config
    inputs: Inputs

    def top_result(self, mapping_results: list[MappingResult]):
        """
        apply deepom on om2seq candidate mapping results, and return the best one by deepom score

        :param mapping_results:
            assume that all mapping results are from the same query
        """
        qry = mapping_results[0].qry
        eval_deepom = EvalDeepOM(
            verbose=False,
            localizer=self.inputs.localizer,
            aligner=self.inputs.aligner,
            references={
                str(i): RefSegment(**dict(mapping_result.ref),
                                   margin=self.config.margin,
                                   references=self.inputs.references).ref_segment
                for i, mapping_result in enumerate(mapping_results)
            },
            crops=self.inputs.crops,
        )
        localized = eval_deepom.deepom_localize(dict(qry))
        if len(localized.localizations) < 2:
            index = 0
        else:
            alignment = eval_deepom.top_alignment(crop=localized)
            index = int(alignment.reference_id)
        return mapping_results[index]


if __name__ == '__main__':
    Fire(Benchmark)
