from typing import TextIO

import more_itertools
import numpy as np
import pandas as pd
from utils.pyutils import PydanticClass, PydanticClassConfig, PydanticClassInputs

from utils.pyutils import NDArray


class BNXParser:
    LINES_PER_RECORD = 7
    RUN_DATA = "# Run Data"

    class BNXRun(PydanticClass):
        RunId: str
        SourceFolder: str
        InstrumentSerial: str
        Time: str
        NanoChannelPixelsPerScan: str
        StretchFactor: float
        BasesPerPixel: str
        NumberofScans: int
        ChipId: str
        FlowCell: str
        SNRFilterType: str
        MinMoleculeLength: float
        MinLabelSNR1: float
        MinLabelSNR2: float

        Scan: str = None
        Bank: str = None
        Cohort: str = None

    class BNXRecord(PydanticClass):
        BNXLocalizations: NDArray

        #  BNX file format:
        AvgIntensity: float
        ChipId: str
        Column: int
        EndFOV: int
        EndX: int
        EndY: int
        Flowcell: str
        GlobalScanNumber: int
        LabelChannel: int
        Length: float
        MoleculeID: int
        NumberofLabels: int
        OriginalMoleculeId: int
        RunId: str
        SNR: float
        ScanDirection: str
        ScanNumber: int
        StartFOV: int
        StartX: int
        StartY: int

    def parse_record(self, record: list[str]):
        metadata, data = [line.strip().split() for line in record[:2]]
        return self.BNXRecord(
            BNXLocalizations=np.asarray(data)[1:].astype(float),
            **dict(zip(self.bnx_columns, metadata)),
        )

    def iter_raw_records(self, text_io: TextIO):
        yield from more_itertools.chunked(self._lines(text_io), self.LINES_PER_RECORD)

    def parse_runs(self, text_io: TextIO) -> dict:
        next(iter(self._lines(text_io)))
        return pd.DataFrame(self.run_data_lines, columns=self.run_columns).to_dict()

    def _lines(self, text_io: TextIO):
        self.run_data_lines = []
        for line in text_io:
            name, *data = line.strip().split()
            if name.startswith("#"):
                if line.startswith(self.RUN_DATA):
                    self.run_data_lines.append(line.lstrip(self.RUN_DATA).strip().split('\t'))
                elif name == "#rh":
                    self.run_columns = data
                elif name == "#0h":
                    self.bnx_columns = data
                elif name == "#0f":
                    self.bnx_dtypes = data
            else:
                self.dtypes = dict(zip(self.bnx_columns, self.bnx_dtypes))
                assert self.run_columns is not None
                assert self.bnx_columns is not None
                assert self.bnx_dtypes is not None
                yield line
                yield from text_io
