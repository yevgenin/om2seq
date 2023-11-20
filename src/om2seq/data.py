import itertools
from typing import Optional

import numpy as np
from datasets import Dataset
from fire import Fire
from scipy.stats import loguniform

from utils.pyutils import PydanticClass, PydanticClassConfig, PydanticClassInputs
from torch.utils import data
from tqdm import tqdm

from deepom.aligner import DeepOMAligner, Orientation
from om2seq.cropping import Cropper
from om2seq.env import ENV
from deepom.image_preprocessing import ImagePreprocessor
from om2seq.plotting import CropPlot
from utils.alignment import AlignedImage
from utils.bnx_images import ImageReader
from utils.bnx_parse import BNXParser
from utils.gcs_utils import any_file_open
from utils.genome import GenomePatternMapper
from utils.dataset_tasks import DatasetTask, ParallelTask, BaseTask
from utils.xmap_parse import XMAPParser, XMAPOrientation


class Error(PydanticClass):
    error: str


class GenomeDataset(DatasetTask, ParallelTask):
    _task_name = 'genome'

    class Config(DatasetTask.Config, ParallelTask.Config):
        fasta_file: str = ENV.FASTA_FILE
        pattern: str = ENV.PATTERN

    config: Config

    @property
    def version_path(self):
        return super().version_path / f'pattern={self.config.pattern}'

    def create_dataset(self):
        with any_file_open(self.config.fasta_file) as f:
            genome = GenomePatternMapper(pattern=self.config.pattern)
            ds = Dataset.from_list(list(genome.sequence_records(f)))
            return self._thread_map(func=genome.find_pattern, ds=ds)

    def references(self):
        return {
            ref.id: ref.positions for ref in
            (GenomePatternMapper.Output(**ref) for ref in self.dataset().with_format('numpy'))
        }


class BNXDataset(DatasetTask):
    _task_name = 'bnx'

    class Config(DatasetTask.Config):
        bnx_file: str = ENV.BNX_FILE

    config: Config

    def create_dataset(self):
        with any_file_open(self.config.bnx_file) as f:
            bnx_parser = BNXParser()
            items = bnx_parser.iter_raw_records(f)
            if self.config.limit is not None:
                items = itertools.islice(items, self.config.limit)
            return Dataset.from_list([
                dict(bnx_parser.parse_record(item)) for item in tqdm(items, desc='Parsing BNX')
            ])

    def bnx_runs(self):
        with any_file_open(self.config.bnx_file) as f:
            return BNXParser().parse_runs(f)


class XMAPDataset(DatasetTask):
    _task_name = 'xmap'

    class Config(DatasetTask.Config):
        file: str = ENV.XMAP_FILE

    def create_dataset(self):
        with any_file_open(self.config.file) as f:
            items = XMAPParser().parse(f)
            if self.config.limit is not None:
                items = itertools.islice(items, self.config.limit)
            return Dataset.from_list(list(tqdm(items)))


class AlignedImagesDataset(DatasetTask, ParallelTask):
    _task_name = 'aligned'

    class Config(DatasetTask.Config, ParallelTask.Config):
        nominal_scale: float = ENV.NOMINAL_SCALE
        bnx_scale: float = ENV.BNX_SCALE
        limit: int = 100000

    def create_dataset(self):
        with self.wandb_init():
            self.bnx = BNXDataset()
            self.xmap = XMAPDataset()
            self.references = GenomeDataset().references()
            self.aligner = DeepOMAligner()
            self.image_reader = ImageReader(runs_df=self.bnx.bnx_runs())
            self.image_preprocessor = ImagePreprocessor()

            records = self._bnx_xmap_merged_df().sort_values('Confidence', ascending=False)[:self.config.limit].to_dict(
                orient='records')
            return self._thread_map(self._dataset_item, records)

    def _dataset_item(self, bnx_xmap_record: dict):
        try:
            bnx_record = BNXParser.BNXRecord(**bnx_xmap_record)
            image = self.image_reader.read_image(bnx_record).image
            image = self.image_preprocessor.preprocess_image(image)
            return AlignedImage(
                image=image,
                **self._align_xmap(
                    bnx_record=bnx_record,
                    xmap_record=XMAPParser.XMAPRecord(**bnx_xmap_record)
                ).model_dump()
            ).model_dump()
        except Exception as err:
            if self.config.raise_errors:
                raise
            return Error(error=str(err)).model_dump()

    def _align_xmap(self, bnx_record: BNXParser.BNXRecord,
                    xmap_record: XMAPParser.XMAPRecord) -> DeepOMAligner.Alignment:
        orientation = Orientation[XMAPOrientation(str(xmap_record.Orientation)).name]
        reference_id = self._ref_id_from_xmap(xmap_record.RefContigID)
        return self.aligner.align_to_ref(
            self.aligner.QueryToRef(
                query_positions=bnx_record.BNXLocalizations[:-1],
                query_scale=self.config.nominal_scale / self.config.bnx_scale,
                orientation=orientation,
                reference_id=reference_id
            ),
            reference=self.references[reference_id],
        )

    def _ref_id_from_xmap(self, xmap_ref_id: int):
        for ref in self.references:
            if ref.startswith(f'NC_0000{xmap_ref_id:02d}'):
                return ref
        assert False

    def _bnx_xmap_merged_df(self):
        bnxdf = self.bnx.dataset().to_pandas()
        xmapdf = self.xmap.dataset().to_pandas()

        df = xmapdf.merge(bnxdf.drop(columns='LabelChannel'), left_on='QryContigID', right_on='MoleculeID')
        df = df[df['StartFOV'] == df['EndFOV']]
        return df


class TrainingSplit(DatasetTask):
    _task_name = 'split'

    class Config(AlignedImagesDataset.Config):
        eval_dataset_size: int = 1000
        test_dataset_size: int = 1000
        subset_seed: int = ENV.SEED

    def create_dataset(self) -> Dataset:
        df = self.aligned_dataset().to_pandas()
        ids = set(df['MoleculeID'].unique())
        rng = np.random.default_rng(self.config.subset_seed)
        eval_molecules = rng.choice(list(ids), size=self.config.eval_dataset_size, replace=False)
        test_molecules = rng.choice(list(ids - set(eval_molecules)), size=self.config.test_dataset_size, replace=False)

        df['subset'] = df['MoleculeID'].apply(
            lambda x: 'eval' if x in eval_molecules else 'test' if x in test_molecules else 'train'
        )
        return Dataset.from_pandas(df[['MoleculeID', 'subset']], preserve_index=False)

    def aligned_dataset(self):
        return AlignedImagesDataset(**dict(self.config)).dataset()

    def dataset(self) -> Dataset:
        df_subset = super().dataset().to_pandas()
        df = self.aligned_dataset().to_pandas()
        assert (df['MoleculeID'] == df_subset['MoleculeID']).all()
        df['subset'] = df_subset['subset']
        # df = df.merge(ds.to_pandas(), on='MoleculeID')
        return Dataset.from_pandas(df, preserve_index=False)

    def dataset_dict(self):
        df = self.dataset().to_pandas()
        df_dict = {
            'train': df[df['subset'] == 'train'],
            'eval': df[df['subset'] == 'eval'],
            'test': df[df['subset'] == 'test'],
        }
        return {
            key: Dataset.from_pandas(df, preserve_index=False)
            for key, df in df_dict.items()
        }


class TrainingDataset(data.Dataset, BaseTask):
    class Config(Cropper.Config, BaseTask.Config):
        qry_len: Optional[int] = None
        ref_len: int = 200 * 1000
        qry_batch_len: int = ref_len

        eval_dataset_size: int = 1000
        min_len: int = 30 * 1000
        max_len: int = qry_batch_len
        crops_seed: int = ENV.SEED
        sample_seed: int = ENV.SEED
        aligned_limit: int = AlignedImagesDataset.Config().limit

    def __init__(self, references: dict = None, **kwargs):
        super().__init__(**kwargs)
        if references is None:
            references = GenomeDataset().references()
        self.references = references
        self.crops_rng = np.random.default_rng(self.config.crops_seed)

        self.training_split = TrainingSplit(limit=self.config.aligned_limit).dataset_dict()
        self.train_subset = self.training_split['train']
        self.crops_eval_dataset = self.generate_crops_dataset(aligned_images_dataset=self.training_split['eval'])

    def generate_crops_dataset(self, aligned_images_dataset: Dataset, qry_len: int = None, limit: int = None):
        return Dataset.from_list(list(itertools.islice(
            (self._crop(aligned_image=aligned_image, qry_len=qry_len)
             for aligned_image in aligned_images_dataset
             if AlignedImage(**aligned_image).reference_id in self.references),
            limit
        )))

    def __getitem__(self, index: int):
        return self._crop(aligned_image=self.train_subset[int(index)], qry_len=self.config.qry_len)

    def __len__(self):
        return len(self.train_subset)

    def _crop(self, aligned_image: dict, qry_len: int = None):
        if qry_len is None:
            qry_len = self._length_sample()
        return Cropper(**self.config.model_dump() | dict(qry_len=qry_len),
                       references=self.references, rng=self.crops_rng,
                       ).random_crop(
            AlignedImage(**aligned_image)).model_dump() | aligned_image

    def _length_sample(self):
        if self.config.min_len == self.config.max_len:
            return self.config.min_len

        return int(loguniform.rvs(self.config.min_len, self.config.max_len))


class Datasets:
    def create(self, **kwargs):
        TrainingSplit(**kwargs).dataset()

    def upload(self, **kwargs):
        AlignedImagesDataset(**kwargs).upload_to_cloud()


if __name__ == '__main__':
    Fire(Datasets)
