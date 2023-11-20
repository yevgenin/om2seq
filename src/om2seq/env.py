import multiprocessing
from pathlib import Path
from typing import Literal, Optional

import joblib
import torch
from cloudpathlib import AnyPath
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Env(BaseSettings):
    LOCAL_OUT_DIR: str
    LOCAL_DATA_DIR: str
    USE_CLOUD_DATA: bool = True
    CLOUD_OUT_DIR: str
    CLOUD_DATA_DIR: str
    CACHE_DIR: str = 'out/cache'
    WANDB_SWEEP_ID_FILE: str
    BIONANO_IMAGES_DIR: str
    NUM_PROC: int = multiprocessing.cpu_count() - 1
    NUM_THREADS: int = multiprocessing.cpu_count()
    BNX_FILES: dict = {
        'CTTAAG': ["T1_chip2_channels_swapped.bnx"],
        'GCTCTTC': [
            "GM12878_mtaq_ecodam_180319_KXPTQGOLPQHGFNWU_F1P1_3_19_2019_5_36_52_AM_RawMolecules.bnx",
            "GM12878_mtaq_ecodam_180319_KXPTQGOLPQHGFNWU_F1P1_3_20_2019_6_47_37_AM_RawMolecules.bnx",
            "GM12878_mtaq_ecodam_180319_KXPTQGOLPQHGFNWU_F1P1_3_22_2019_6_50_00_AM_RawMolecules.bnx",
        ]
    }
    DATASET_PARQUET: str = 'dataset.parquet'

    CONFIG_JSON: str = 'config.json'
    PATTERN: Literal['CTTAAG', 'GCTCTTC'] = 'CTTAAG'
    BNX_FILE: str = BNX_FILES[PATTERN][0]
    XMAP_FILE: str = 'exp_refineFinal1.xmap'
    FASTA_FILE: str = 'GCF_000001405.40_GRCh38.p14_genomic.fna'
    DEEPOM_MODEL_FILE: str = 'src/deepom/deepom.pt'
    BNX_SCALE: float = 375
    NOMINAL_SCALE: float = 320
    GCS_BLOB_TIMEOUT: float = 60 * 60 * 24 * 7  # 7 days
    GCS_BLOB_NUM_RETRIES: int = 2
    SEED: int = 0
    FIGURES_DIR: Path

    def __init__(self, **kw):
        super().__init__(**kw)

        if self.USE_CLOUD_DATA:
            data_root = AnyPath(self.CLOUD_DATA_DIR)
        else:
            data_root = AnyPath(self.LOCAL_DATA_DIR)

        self.BNX_FILE = str(data_root / self.BNX_FILE)
        self.FASTA_FILE = str(data_root / self.FASTA_FILE)
        self.XMAP_FILE = str(data_root / self.XMAP_FILE)


ENV = Env()

joblib_memory = joblib.Memory(ENV.CACHE_DIR, verbose=0)
