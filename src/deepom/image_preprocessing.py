import numpy as np
from devtools import debug
from monai.transforms import Compose, SpatialPad, SpatialCrop, ScaleIntensity, SqueezeDim, \
    EnsureChannelFirst
from monai.utils import Method
from utils.pyutils import PydanticClass, PydanticClassConfig, PydanticClassInputs
from typing import Optional


class ImagePreprocessor:
    class Config(PydanticClassConfig):
        image_width: int = 5
        pad_or_crop_length: Optional[int] = None
        image_channel: int = 0

    def __init__(self, **config):
        self.config = self.Config(**config)
        debug(self.config)

    def preprocess_image(self, image: np.ndarray):
        assert image.ndim == 3
        image = image[self.config.image_channel]

        target_width = self.config.image_width
        source_width = image.shape[0] // 2 + 1

        image = image[source_width - target_width // 2: source_width + target_width // 2 + 1, :]

        return Compose([
            EnsureChannelFirst(channel_dim='no_channel'),
            ScaleIntensity(),
            self.pad_or_crop((self.config.image_width, self.config.pad_or_crop_length)),
            SqueezeDim(),
        ])(image)

    @staticmethod
    def pad_or_crop(size):
        return Compose([
            SpatialPad(spatial_size=size, method=Method.END),
            SpatialCrop(roi_slices=[slice(0, s) for s in size]),
        ])
