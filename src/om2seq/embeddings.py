import numpy as np
from datasets import Dataset
from devtools import debug
from faiss import METRIC_INNER_PRODUCT

from om2seq.mapping import RefEmb
from utils.pyutils import PydanticClassInputs
from tqdm import tqdm
from typing import Optional
from om2seq.inference import InferenceModel
from om2seq.data import TrainingDataset
from utils.dataset_tasks import DatasetTask

default_training_dataset_config = TrainingDataset.Config()


class EmbeddingsDB(DatasetTask):
    _task_name = 'ref_emb'
    INDEX_NAME = 'ref_emb'

    class Config(DatasetTask.Config):
        model_id: str
        bin_size: int = default_training_dataset_config.bin_size
        ref_len: int = default_training_dataset_config.ref_len
        ref_step: int = 30000
        faiss_device: Optional[int] = None  # 0 if torch.cuda.is_available() else None
        faiss_string_factory: str = None  # 'HNSW32'

    class Inputs(PydanticClassInputs):
        references: dict
        inference_model: InferenceModel

    config: Config
    inputs: Inputs

    @property
    def version_path(self):
        return (super().version_path /
                f'model_id={self.config.model_id}' /
                f'ref_len={self.config.ref_len}' /
                f'ref_step={self.config.ref_step}'
                )

    def build(self):
        self.create_and_upload()

    def create_dataset(self):
        ds = Dataset.from_list(
            [_.model_dump() for _ in tqdm(self._iter_genome_embs(), desc=self.create_dataset.__name__)]
        )
        ds.add_faiss_index(column=self.INDEX_NAME, metric_type=METRIC_INNER_PRODUCT,
                           device=self.config.faiss_device,
                           string_factory=self.config.faiss_string_factory, faiss_verbose=False)
        return ds

    def _save_local(self, ds: Dataset):
        super()._save_local(ds)
        self._save_index(ds)

    @property
    def index_file(self):
        return self.local_file.with_name(f'{self.INDEX_NAME}.index')

    @property
    def index_cloud_file(self):
        return self.cloud_file.with_name(f'{self.INDEX_NAME}.index')

    def load(self):
        ds = super().load()
        if not self.index_file.exists():
            print(f'Index file {self.index_file} does not exist')
            if self.index_cloud_file.exists():
                print(f'Downloading index from {self.index_cloud_file}')
                self._download(cloud_src=self.index_cloud_file, local_dst=self.index_file)
            else:
                with debug.timer(ds.add_faiss_index.__name__):
                    self._save_index(ds)

                self._upload(local_src=self.index_file, cloud_dst=self.index_cloud_file)

        with debug.timer(ds.load_faiss_index.__name__):
            print(f'Loading index from {self.index_file}')
            ds.load_faiss_index(index_name=self.INDEX_NAME, file=self.index_file, device=self.config.faiss_device)
        return ds

    def upload_to_cloud(self):
        super().upload_to_cloud()
        self._upload(local_src=self.index_file, cloud_dst=self.index_cloud_file)

    def _save_index(self, ds):
        print('Saving index: ', self.index_file)
        ds.save_faiss_index(index_name=self.INDEX_NAME, file=self.index_file)

    def _iter_genome_embs(self):
        for reference_id, reference in tqdm(self.inputs.references.items(),
                                            desc=self._iter_genome_embs.__name__):
            if max(reference) < self.config.ref_len:
                continue

            yield from self.compute_reference_embeddings(
                reference=reference,
                reference_id=reference_id,
                bin_size=self.config.bin_size,
                ref_len=self.config.ref_len,
                ref_step=self.config.ref_step,
            )

    def compute_reference_embeddings(self, reference, reference_id, bin_size, ref_len, ref_step):
        emb_step_bins = ref_step // bin_size
        ref_binned = np.bincount((reference / bin_size).astype(int))
        window_length = ref_len // bin_size
        x_embs = self.inputs.inference_model.x_embs_rolling(ref_binned, window_length=window_length,
                                                            step=emb_step_bins,
                                                            limit=self.config.limit)
        for i, x_emb in enumerate(x_embs):
            ref_start = i * emb_step_bins * bin_size
            yield RefEmb(
                ref_emb=x_emb,
                reference_id=str(reference_id),
                ref_start=ref_start,
                ref_stop=ref_start + window_length * bin_size,
            )
