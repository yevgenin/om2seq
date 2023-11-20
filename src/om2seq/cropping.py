import numpy as np
from devtools import debug

from deepom.aligner import Orientation
from utils.pyutils import PydanticClass, PydanticClassConfig, PydanticClassInputs
from om2seq.env import ENV
from utils.alignment import AlignmentInfo, AlignedImage
from utils.pyutils import NDArray


class Cropper:
    class Config(PydanticClassConfig):
        qry_len: int
        qry_batch_len: int
        ref_len: int
        ref_dtype: str = 'uint8'
        image_scale: float = ENV.NOMINAL_SCALE
        bin_size: int = int(image_scale)
        locs_scale: float = ENV.BNX_SCALE
        add_random_shift: bool = True
        add_random_flip: bool = True

    class CropRef(PydanticClass):
        x: NDArray
        ref_start: int
        ref_stop: int
        crop_ref: NDArray

    class CropQuery(PydanticClass):
        y: NDArray
        qry_start: float
        qry_stop: float
        crop_image: NDArray
        pad_amount: int
        crop_orientation: Orientation

    class AlignedCrop(CropRef, CropQuery):
        image_scale: float
        bin_size: float

    config: Config

    def __init__(self, references: dict[str, np.ndarray], rng=None, **config):
        self.config = self.Config(**config)
        self.references = references
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def random_crop(self, aligned_image: AlignedImage):
        ref = self._make_ref_crop(aligned_image)
        qry = self._make_qry_crop(aligned_image, ref_center=self._ref_center(ref))

        return self.AlignedCrop(
            image_scale=self.config.image_scale,
            bin_size=self.config.bin_size,
            **ref.model_dump(),
            **qry.model_dump(),
        )

    def _ref_center(self, ref: CropRef):
        center = (ref.ref_start + ref.ref_stop) / 2
        if self.config.add_random_shift:
            size = self.config.qry_len
            return self.rng.integers(center - size // 2, center + size // 2)
        else:
            return center

    def _make_ref_crop(self, aligned_image: AlignedImage):
        alignment_info = AlignmentInfo(**dict(aligned_image), references=self.references)
        parent_lims = alignment_info.ref_lims
        i = parent_lims[0]
        j = parent_lims[1] - self.config.ref_len
        if i < j:
            ref_start = self.rng.integers(i, j)
        else:
            ref_start = i
        ref_stop = ref_start + self.config.ref_len
        crop_ref = alignment_info.reference_segment(ref_start, ref_stop)

        return self.create_crop_ref(crop_ref, ref_start, ref_stop)

    def create_crop_ref(self, crop_ref, ref_start, ref_stop):
        ref_num_bins = self.config.ref_len // self.config.bin_size
        x = np.bincount(((crop_ref - ref_start) / self.config.bin_size).astype(int), minlength=ref_num_bins)
        x = x[:ref_num_bins].astype(self.config.ref_dtype)
        assert len(x.shape) == 1
        assert x.shape[0] == ref_num_bins
        return self.CropRef(
            x=x,
            ref_start=ref_start,
            ref_stop=ref_stop,
            crop_ref=crop_ref,
        )

    def _make_qry_crop(self, aligned_image: AlignedImage, ref_center: float):
        alignment_info = AlignmentInfo(**aligned_image.model_dump(), references=self.references)
        qry_center = int(alignment_info.transform_to_query(ref_center).item() / self.config.locs_scale)

        image_len = int(self.config.qry_len / self.config.image_scale) // 2 * 2
        image_batch_len = int(self.config.qry_batch_len / self.config.image_scale) // 2 * 2

        assert image_len <= image_batch_len

        qry_start = qry_center - image_len // 2
        qry_stop = qry_start + image_len
        crop_image = aligned_image.image[:, qry_start:qry_stop]

        if self.config.add_random_flip and self.rng.random() < 0.5:
            crop_image = crop_image[:, ::-1]
            crop_orientation = Orientation.REVERSE
        else:
            crop_orientation = Orientation.FORWARD

        #  pad to image_batch_len
        len_delta = image_batch_len - crop_image.shape[-1]
        pad_amount = len_delta // 2
        y = np.pad(crop_image, ((0, 0), (pad_amount, len_delta - pad_amount)), mode='constant')

        assert len(y.shape) == 2
        assert y.shape[1] == image_batch_len, f'{y.shape[1]} != {image_batch_len}'
        return self.CropQuery(
            y=y,
            qry_start=qry_start,
            qry_stop=qry_stop,
            crop_image=crop_image,
            pad_amount=pad_amount,
            crop_orientation=crop_orientation,
        )
