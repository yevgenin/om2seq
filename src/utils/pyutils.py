from typing import Annotated, Optional

import numpy as np
import torch
from pydantic import BaseModel, PlainValidator

NDArray = Annotated[np.ndarray, PlainValidator(np.asarray)]
TorchTensor = Annotated[torch.Tensor, PlainValidator(torch.as_tensor)]


class PydanticClass(BaseModel, protected_namespaces=(), use_enum_values=True):
    pass


class PydanticClassConfig(PydanticClass):
    version: Optional[str] = None
    verbose: bool = True
    wandb_enabled: bool = True


class PydanticClassInputs(PydanticClass, arbitrary_types_allowed=True):
    pass
