import itertools
import shutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Iterable

import wandb
from cloudpathlib import AnyPath
from datasets import Dataset
from devtools import debug
from utils.pyutils import PydanticClass, PydanticClassConfig, PydanticClassInputs
from tqdm.auto import tqdm

from om2seq.env import ENV
from utils.gcs_utils import gcs_blob
from typing import Optional


class BaseTask:
    Config = PydanticClassConfig

    def __init__(self, **kwargs):
        assert issubclass(self.Config, PydanticClassConfig)
        self.config = self.Config(**kwargs)
        if hasattr(self, 'Inputs'):
            self.inputs = self.Inputs(**kwargs)

        if self.config.verbose:
            debug(name=self.__class__.__name__, **dict(self.config))

    def wandb_init(self):
        return wandb.init(mode='disabled' if not self.config.wandb_enabled else None, tags=[self.__class__.__name__])


class ParallelTask(BaseTask):
    class Config(BaseTask.Config):
        num_threads: int = ENV.NUM_THREADS

    config: Config

    def _thread_map(self, func: Callable, ds: Dataset | list, limit: int = None):
        if limit is None:
            limit = self.config.limit

        if isinstance(ds, list):
            ds = Dataset.from_list(ds)

        if limit is not None:
            ds = ds.select(range(limit))

        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            items = tqdm(
                executor.map(lambda x: func(x) | x, ds),
                total=len(ds), desc=self.__class__.__name__
            )
            return Dataset.from_list(list(items))


class MPTask(ParallelTask):
    class Config(PydanticClassConfig):
        num_proc: int = ENV.NUM_PROC

    config: Config

    def map_ds(self, func: Callable, ds: Dataset, limit: int = None):
        if limit is None:
            limit = self.config.limit

        if limit is not None:
            ds = ds.select(range(limit))

        return ds.map(func, num_proc=self.config.num_proc, keep_in_memory=True)

    def _map_pool(self, ds, func, limit):
        with Pool(processes=self.config.num_proc) as pool:
            x = pool.imap_unordered(func, ds)
            x = tqdm(x, total=limit)
            x = list(x)
            return Dataset.from_list(x)

    def map_iter(self, func: Callable, items: Iterable, limit: int = None):
        if limit is None:
            limit = self.config.limit

        if limit is not None:
            items = itertools.islice(items, limit)

        input_name = 'input_item'
        items = ({input_name: _} for _ in items)
        items = list(tqdm(items, desc=self.__class__.__name__, total=limit))
        ds = Dataset.from_list(items)
        ds = ds.map(
            lambda row: func(row[input_name]),
            remove_columns=[input_name],
            num_proc=self.config.num_proc,
            keep_in_memory=True,
        )
        return ds


class DatasetTask(BaseTask):
    _task_name = None

    class Config(PydanticClassConfig):
        local_dir: str = ENV.LOCAL_OUT_DIR
        cloud_dir: str = ENV.CLOUD_OUT_DIR
        limit: Optional[int] = None
        raise_errors: bool = False

    config: Config

    @property
    def task_name(self):
        if self._task_name is None:
            return self.__class__.__name__
        return self._task_name

    @property
    def version_path(self):
        return Path(f'task={self.task_name}', f'limit={self.config.limit}')

    def remove_version(self):
        local_file = self.local_file
        if local_file.exists():
            shutil.rmtree(local_file.parent)

        cloud_file = self.cloud_file

        if cloud_file.exists():
            cloud_file.parent.rmtree()

    def get_cloud_file_path(self, path: str):
        return AnyPath(self.config.cloud_dir) / self.version_path / Path(path)

    def get_local_file_path(self, path: str):
        return AnyPath(self.config.local_dir) / self.version_path / Path(path)

    @property
    def cloud_file(self):
        if self.config.cloud_dir is None:
            return None
        return AnyPath(self.config.cloud_dir) / self.version_path / ENV.DATASET_PARQUET

    @property
    def local_file(self):
        return AnyPath(self.config.local_dir) / self.version_path / ENV.DATASET_PARQUET

    def dataset(self) -> Dataset:
        if self.local_file.exists() or self.cloud_file.exists():
            ds = self.load()
        else:
            ds = self.create_and_upload()
        return ds

    def load(self) -> Dataset:
        self._download(cloud_src=self.cloud_file, local_dst=self.local_file)
        return debug(self.load_from_file(self.local_file))

    def create_and_upload(self) -> Dataset:
        ds = self.create_and_save()
        self.upload_to_cloud()
        return debug(ds)

    def upload_to_cloud(self):
        with debug.timer(self.upload_to_cloud.__name__):
            if self.cloud_file:
                self._upload(self.local_file, self.cloud_file)
                self.cloud_file.with_name(ENV.CONFIG_JSON).write_text(self.config.model_dump_json())

    def _upload(self, local_src: Path, cloud_dst: AnyPath):
        debug(self, local_src, cloud_dst)
        cloud_dst.parent.mkdir(parents=True, exist_ok=True)
        with local_src.open('rb') as in_file:
            total_bytes = local_src.stat().st_size
            with tqdm.wrapattr(in_file, "read", total=total_bytes, miniters=1, desc=self._upload.__name__) as file_obj:
                gcs_blob(str(cloud_dst)).upload_from_file(
                    file_obj,
                    size=total_bytes,
                    timeout=ENV.GCS_BLOB_TIMEOUT,
                    num_retries=ENV.GCS_BLOB_NUM_RETRIES
                )

    def _download(self, cloud_src: AnyPath, local_dst: Path):
        debug(self, cloud_src, local_dst)
        if not local_dst.exists() and cloud_src:
            if not cloud_src.exists():
                print('cloud file does not exist: ', debug.format(cloud_src))
                return
            local_dst.parent.mkdir(parents=True, exist_ok=True)
            with local_dst.open('wb') as out_file:
                total_bytes = gcs_blob(str(cloud_src)).size
                with tqdm.wrapattr(out_file, "write", total=total_bytes, miniters=1,
                                   desc=self._download.__name__) as file_obj:
                    gcs_blob(str(cloud_src)).download_to_file(
                        file_obj,
                        timeout=ENV.GCS_BLOB_TIMEOUT,
                    )

    def create_and_save(self):
        debug(self)
        ds = self.create_dataset()
        self._save_local(ds)
        return ds

    def _save_local(self, ds: Dataset):
        local_file = self.local_file
        debug(self, local_file)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        with debug.timer(self._save_local.__name__):
            ds.to_parquet(str(local_file))

    def load_from_file(self, local_file):
        debug(self, local_file)
        return Dataset.from_parquet(str(local_file), keep_in_memory=True, cache_dir=None)

    def create_dataset(self) -> Dataset:
        raise NotImplementedError
