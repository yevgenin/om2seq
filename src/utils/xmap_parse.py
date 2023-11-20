from enum import Enum
from typing import TextIO

from utils.pyutils import PydanticClass, PydanticClassConfig, PydanticClassInputs


class XMAPOrientation(Enum):
    FORWARD = '+'
    REVERSE = '-'


class XMAPParser:
    class XMAPRecord(PydanticClass):
        XmapEntryID: int
        QryContigID: int
        RefContigID: int
        QryStartPos: float
        QryEndPos: float
        RefStartPos: float
        RefEndPos: float
        Orientation: XMAPOrientation
        Confidence: float
        HitEnum: str
        QryLen: float
        RefLen: float
        LabelChannel: int
        Alignment: str
        MapWt: float

    def __init__(self):
        self.dtypes = None
        self.columns = None

    def parse_line(self, line: str):
        data = dict(zip(self.columns, line.strip().split()))
        return self.XMAPRecord(**data).model_dump()

    def parse(self, text_io: TextIO):
        yield from map(self.parse_line, self.lines(text_io))

    def lines(self, text_io: TextIO):
        for line in text_io:
            name, *data = line.strip().split()
            if name.startswith("#"):
                if name == "#h":
                    self.columns = data
                elif name == "#f":
                    self.dtypes = data
            else:
                self.dtypes = dict(zip(self.columns, self.dtypes))
                assert self.columns is not None
                assert self.dtypes is not None
                yield line
                yield from text_io
