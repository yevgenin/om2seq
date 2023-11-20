from typing import NamedTuple, Union

import numpy as np
import torch
from devtools import debug
from monai.networks.nets import BasicUNet
from monai.transforms import Compose, SpatialPad, SqueezeDim, \
    DivisiblePad, ToNumpy
from monai.utils import Method, convert_to_tensor
from utils.pyutils import PydanticClass, PydanticClassConfig, PydanticClassInputs

from deepom.image_preprocessing import ImagePreprocessor
from utils.env import ENV
from utils.pyutils import NDArray


class LocalizerEnum:
    BG = 0
    STRAY = 1
    FG = 2


class LocalizerOutputs(NamedTuple):
    label_bg: Union[np.ndarray, torch.Tensor]
    label_stray: Union[np.ndarray, torch.Tensor]
    label_fg: Union[np.ndarray, torch.Tensor]
    loc_output: Union[np.ndarray, torch.Tensor]


class DeepOMLocalizer(ImagePreprocessor):
    class Config(ImagePreprocessor.Config):
        divisible_size: int = 16
        min_spatial_size: int = 32
        upsample: str = "pixelshuffle"
        unet_channel_divider: int = 1
        out_channels: int = len(LocalizerOutputs.__annotations__)
        model_file: str = ENV.DEEPOM_MODEL_FILE
        device: str = 'cpu'
        unet_features: int = [32, 32, 64, 128, 256, 32]

    class LocalizerOutput(PydanticClass):
        localizations: NDArray = None
        preprocessed_image: NDArray = None

        offset_pred: NDArray = None
        output_tensor: NDArray = None
        occupancy_pred: NDArray = None
        outputs: NDArray = None

    dtype = torch.float32

    def __init__(self, **config):
        super().__init__(**config)
        self.module = self._module_build()
        self.module.load_state_dict(torch.load(self.config.model_file, map_location=self.config.device))
        self.module.to(self.config.device)
        self.module.eval()

    def inference(self, image: NDArray, preprocess_image: bool = True, extras=False):
        if preprocess_image:
            preprocessed_image = self.preprocess_image(image)
        else:
            preprocessed_image = image
        output_tensor = self._inference_forward_pass(image)

        outputs = LocalizerOutputs(*output_tensor)
        occupancy_pred = np.stack([
            outputs.label_bg,
            outputs.label_stray,
            outputs.label_fg,
        ]).argmax(axis=0)
        output = outputs.loc_output

        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output)

        offset_pred = torch.sigmoid(output).numpy()
        predicted_fg_indices = np.flatnonzero(occupancy_pred == LocalizerEnum.FG)
        localizations = predicted_fg_indices + offset_pred[predicted_fg_indices]

        return self.LocalizerOutput(
            localizations=localizations,
            preprocessed_image=preprocessed_image if extras else None,
            output_tensor=output_tensor if extras else None,
            occupancy_pred=occupancy_pred if extras else None,
            offset_pred=offset_pred if extras else None,
        )

    def _to_tensor(self, obj):
        return convert_to_tensor(obj, device=self.config.device, dtype=self.dtype)

    def _module_forward(self, module_input: np.ndarray):
        return Compose([
            self._to_tensor,
            self.module,
        ])(module_input)

    def _inference_forward_pass(self, module_input: np.ndarray):
        x = Compose([
            SpatialPad(spatial_size=self.config.min_spatial_size, method=Method.END),
            DivisiblePad(k=self.config.divisible_size, method=Method.END)
        ])(module_input)
        x = self._module_forward(x[None])
        x = Compose([
            ToNumpy(),
            SqueezeDim(),
            self.pad_or_crop(module_input.shape[-1:]),
        ])(x)

        return x

    def _module_build(self):
        return BasicUNet(
            spatial_dims=1,
            features=tuple(np.stack(self.config.unet_features) // self.config.unet_channel_divider),
            in_channels=self.config.image_width,
            out_channels=self.config.out_channels,
            upsample=self.config.upsample,
        )
