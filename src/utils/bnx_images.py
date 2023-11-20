from typing import Optional

import imagecodecs
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath
from devtools import debug
from utils.pyutils import PydanticClass, PydanticClassConfig, PydanticClassInputs

from utils.env import ENV, joblib_memory
from utils.bnx_parse import BNXParser
from utils.gcs_utils import any_file_read_bytes
from utils.pyutils import NDArray
from utils.image_utils import extract_segment_from_endpoints


class ImageReader:
    class Config(PydanticClassConfig):
        bionano_images_dir: str = ENV.BIONANO_IMAGES_DIR
        bnx_channel: int = 3
        fov_size: int = 2048
        segment_width: int = 11

    class BionanoImage(PydanticClass):
        # multi-channel image with dimensions (channels, height, width)
        image: NDArray
        fov_file: str

    def __init__(self, runs_df: dict, **config):
        self.config = self.Config(**config)
        debug(self.config)
        self.runs = self._parse_runs(runs_df)

    def _parse_runs(self, runs_df: dict) -> dict[str, BNXParser.BNXRun]:
        df = pd.DataFrame(runs_df)
        # parse the SourceFolder column into Scan, Bank, Cohort
        df = df.join(df["SourceFolder"].str.extract(r"Cohort(?P<Scan>\d\d)(?P<Bank>\d)(?P<Cohort>\d)"))
        data = df.set_index("RunId", drop=False).T.to_dict()
        # return a dict of BNXRun objects indexed by RunId
        return {key: BNXParser.BNXRun(**value) for key, value in data.items()}

    def read_image(self, bnx_record: BNXParser.BNXRecord):
        fov_file = self._fov_relpath(bnx_record)
        file = AnyPath(self.config.bionano_images_dir) / fov_file
        endpoints = self._parse_segment_endpoints(bnx_record)
        fov = read_jxr_image(str(file))
        image = extract_segment_from_endpoints(fov[None], endpoints=endpoints,
                                               segment_width=self.config.segment_width)
        return self.BionanoImage(
            image=image,
            fov_file=fov_file,
        )

    def _parse_segment_endpoints(self, molecule: BNXParser.BNXRecord):
        start_y, start_x = molecule.StartY, molecule.StartX
        start_y = start_y + (molecule.StartFOV - 1) * self.config.fov_size

        stop_y, stop_x = molecule.EndY, molecule.EndX
        stop_y = stop_y + (molecule.EndFOV - 1) * self.config.fov_size

        return np.stack([
            [start_y, start_x],
            [stop_y, stop_x],
        ])

    def _fov_relpath(self, bnx_record: BNXParser.BNXRecord):
        Channel = self.config.bnx_channel
        bnx_run = self.runs[bnx_record.RunId]
        C_digits = bnx_record.Column
        ChipId = bnx_record.ChipId.split(",")[-2].lstrip("Run_")
        return f"{ChipId}/FC{bnx_record.Flowcell}/Scan{bnx_run.Scan}/Bank{bnx_run.Bank}/B{bnx_run.Bank}_CH{Channel}_C{C_digits:03d}.jxr"


@joblib_memory.cache(mmap_mode='r')
def read_jxr_image(file: str):
    return imagecodecs.jpegxr_decode(any_file_read_bytes(file))
