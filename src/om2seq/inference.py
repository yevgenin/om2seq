import itertools
from typing import Iterable

import more_itertools
import numpy as np
import torch
from devtools import debug

from om2seq.mapping import QryEmb
from utils.pyutils import PydanticClassConfig
from tqdm import tqdm

from om2seq.cropping import Cropper
from utils.wandb_utils import WandbRunData
from om2seq.train import HFModel


class InferenceModel:
    class Config(PydanticClassConfig):
        model_id_wandb_run_name: str
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        use_data_parallel: bool = torch.cuda.device_count() > 1
        batch_size: int = 1024
        normalize: bool = True

    def __init__(self, **config):
        self.config = self.Config(**config)
        debug(self.config)
        self.wandb_run_data = WandbRunData(wandb_run_name=self.config.model_id_wandb_run_name)
        self.model = HFModel(**self.wandb_run_data.run_config)
        self.model.load_state_dict(self.wandb_run_data.state_dict)
        self.model.eval()
        self.model.to(self.config.device)
        if self.config.use_data_parallel and self.config.device != 'cpu':
            print('Using DataParallel')
            self.model.encoder_x = torch.nn.DataParallel(self.model.encoder_x)
            self.model.encoder_y = torch.nn.DataParallel(self.model.encoder_y)

    def x_embs_rolling(self, x: np.ndarray, window_length: int, step: int, limit: int = None):
        with torch.no_grad():
            x = torch.from_numpy(x).to(self.model.device).unfold(0, window_length, step)[:limit]
            batch_size = self.config.batch_size
            if self.config.use_data_parallel:
                batch_size = batch_size * torch.cuda.device_count()

            x_emb = np.concatenate([
                self.inference_x(batch=x[i:i + batch_size])
                for i in tqdm(range(0, len(x), batch_size), desc=self.x_embs_rolling.__name__)
                if i < len(x)
            ])

        return x_emb

    def inference_batched(self, items: Iterable[Cropper.AlignedCrop], qry_limit: int = None):
        with torch.inference_mode():
            return [
                QryEmb(**item, qry_emb=y_emb)
                for chunk in itertools.islice(more_itertools.chunked(items, self.config.batch_size), qry_limit)
                for item, y_emb in zip(
                    tqdm(chunk, desc=self.inference_batched.__name__), self.inference_y(chunk),
                )
            ]

    def inference_x(self, batch: torch.Tensor):
        z = self.model.encoder_forward_pass(encoder=self.model.encoder_x, batch=batch)
        if self.config.normalize:
            z = z / z.norm(dim=-1, keepdim=True)
        return z.cpu().numpy()

    def inference_y(self, chunk: list[dict]):
        z = self.model.encoder_forward_pass(
            encoder=self.model.encoder_y,
            batch=self.model.collate_same_size(items=chunk, which='y').to(self.model.device),
        )
        if self.config.normalize:
            z = z / z.norm(dim=-1, keepdim=True)
        return z.cpu().numpy()
