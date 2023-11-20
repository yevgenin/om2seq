from typing import Optional

from deepom.aligner import DeepOMAligner
from om2seq.cropping import Cropper
from utils.alignment import AlignedImage
from utils.pyutils import PydanticClass, NDArray


class QryEmb(Cropper.AlignedCrop, AlignedImage):
    qry_emb: NDArray


class RefEmb(PydanticClass):
    ref_start: int
    ref_stop: int
    reference_id: str
    ref_emb: NDArray


class MappingResult(PydanticClass):
    qry: QryEmb
    ref: RefEmb
    correct: bool
    overlap: int
    score: float


class RefSegment(PydanticClass):
    ref_start: int
    ref_stop: int
    reference_id: str
    references: dict
    margin: int = 0

    @property
    def ref_segment(self):
        ref = self.references[self.reference_id]
        limits = [self.ref_start - self.margin, self.ref_stop + self.margin]
        i, j = ref.searchsorted(limits)
        return ref[i: j]
