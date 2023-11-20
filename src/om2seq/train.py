import itertools
import os
from pathlib import Path
from typing import Iterable, Literal, Any, Optional

import numpy as np
import pandas as pd
import torch
import transformers
import wandb
from devtools import debug
from fire import Fire

from utils.dataset_tasks import BaseTask
from utils.pyutils import PydanticClass, PydanticClassConfig, PydanticClassInputs
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, TrainingArguments, PreTrainedModel, Wav2Vec2Config, WavLMConfig, \
    EarlyStoppingCallback
from transformers.trainer_utils import EvalLoopOutput, IntervalStrategy, SchedulerType, find_executable_batch_size
from transformers.training_args import OptimizerNames

from om2seq.env import ENV
from om2seq.data import TrainingDataset
from utils.pyutils import TorchTensor

NUM_MODEL_PARAMS = 'num_model_params'


class SweepMinMaxConfig(PydanticClass):
    distribution: Literal['log_uniform_values', 'int_uniform', 'q_log_uniform_values']
    min: Any
    max: Any


class SweepParameters(PydanticClass):
    learning_rate: SweepMinMaxConfig = SweepMinMaxConfig(distribution='log_uniform_values', min=1e-4, max=1e-3)
    model_width_scale: SweepMinMaxConfig = SweepMinMaxConfig(distribution='log_uniform_values', min=.05, max=.5)
    num_hidden_layers: SweepMinMaxConfig = SweepMinMaxConfig(distribution='int_uniform', min=1, max=6)
    # num_attention_heads: SweepMinMaxConfig = SweepMinMaxConfig(distribution='int_uniform', min=6, max=6)
    conv_stride_multiplier: SweepMinMaxConfig = SweepMinMaxConfig(distribution='int_uniform', min=3, max=5)


class SweepConfig(PydanticClassConfig):
    name: str
    method: Literal['bayes'] = 'bayes'
    metric: dict[str, Any] = {'name': 'eval/accuracy', 'goal': 'maximize'}
    early_terminate: dict[str, Any] = {'type': 'hyperband', 'min_iter': 200}
    parameters: SweepParameters = SweepParameters()


def wandb_sweep(version, **config):
    sweep_config = SweepConfig(name=version, **config).model_dump()
    sweep_id = wandb.sweep(debug(sweep_config))
    wandb_sweep_id_file = os.environ['WANDB_SWEEP_ID_FILE']
    Path(debug(wandb_sweep_id_file)).write_text(sweep_id)


DTYPE = torch.float32


class TrainingTasks(BaseTask):
    class Config(TrainingDataset.Config):
        out_dir: str = ENV.LOCAL_OUT_DIR
        version: Optional[str] = None
        sweep_id: Optional[str] = None
        test_mode: bool = False

        #   data
        dataloader_num_workers: int = 0
        x_num_channels: int = 1
        y_num_channels: int = 5

        #   model
        model_config_class: str = WavLMConfig.__name__
        model_conv_stride: int = 1
        model_conv_stride_base: int = 1
        conv_stride_multiplier: int = 3
        model_width_scale: float = .1
        num_hidden_layers: int = 1
        num_attention_heads: int = 6

        #   loss
        normalize: bool = True
        logit_scale_init_value: float = 0.07
        logit_scale_max: float = 100.0

        #   training
        batch_size: Optional[int] = None
        max_steps: int = 200000
        save_steps: int = 1000
        eval_steps: int = 100
        warmup_steps: int = 100
        logging_steps: int = 1
        log_level: str = 'warning'
        limit_eval_batches: Optional[int] = None
        use_cpu: bool = False
        wandb_enabled: bool = True
        starting_batch_size: int = 2 ** 8
        learning_rate: float = 5e-4
        auto_find_batch_size: bool = True
        early_stopping_patience: int = 64
        metric_for_early_stopping: str = 'accuracy'

    def __init__(self, **config_overrides):
        self.config_overrides = config_overrides

        self.config = self.Config(**config_overrides)

        if self.config.test_mode:
            test_config = dict(
                batch_size=2,
                starting_batch_size=2,
                auto_find_batch_size=True,
                max_steps=2,
                limit_eval_batches=1,
                eval_steps=1,
                save_steps=0,
                wandb_enabled=False,
                dataloader_num_workers=0,
                use_cpu=True,
            )
            self.config = self.Config(**(test_config | config_overrides))
        elif self.config.batch_size is not None:
            self.config = self.Config(**dict(self.config) |
                                        dict(starting_batch_size=self.config.batch_size,
                                             auto_find_batch_size=False) | config_overrides
                                      )

    def train(self):
        with self.wandb_init():
            # update config defaults from wandb swipe trial config, overriding with hard config overrides
            self.config = self.Config(**(self.config.model_dump() | dict(wandb.config) | self.config_overrides))

            # show in summary the trial config used for this run
            wandb.summary.update(dict(wandb.config))

            debug(self.config, dict(wandb.config), self.config_overrides)

            if self.config.wandb_enabled:
                wandb.run.log_code(root='src')

            wandb.config.update(self.config.model_dump(), allow_val_change=True)
            dataset = self.dataset()
            find_executable_batch_size(
                lambda batch_size: self.trainer(training_dataset=dataset,
                                                batch_size=debug(batch_size)).train(),
                starting_batch_size=self.config.starting_batch_size,
                auto_find_batch_size=self.config.auto_find_batch_size,
            )()

    def agent(self):
        debug(self.config.sweep_id)
        wandb.agent(sweep_id=self.config.sweep_id, function=self.train)

    def dataset(self):
        return TrainingDataset(**self.config.model_dump())

    def trainer(self, training_dataset: TrainingDataset, **config_overrides):
        trainer = HFModelTrainer(train_dataset=training_dataset, **self.config.model_dump() | config_overrides)
        wandb.summary[NUM_MODEL_PARAMS] = trainer.model.num_parameters()
        wandb.summary.update(dict(train_dataset_size=len(training_dataset)))
        wandb.summary.update(dict(eval_dataset_size=len(training_dataset.crops_eval_dataset)))
        wandb.summary.update(config_overrides)
        return trainer


class HFModel(PreTrainedModel):
    def __init__(self, **trainer_config):
        self.trainer_config = TrainingTasks.Config(**trainer_config)
        self.config_x = self._encoder_config(which='x')
        self.config_y = self._encoder_config(which='y')
        super().__init__(
            config=self.config_x,
        )
        self._init()

    def _init(self):
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * self.trainer_config.logit_scale_init_value)
        self.encoder_x = AutoModel.from_config(self.config_x)
        self.encoder_y = AutoModel.from_config(self.config_y)

    def _encoder_config(self, which: Literal['x', 'y']):
        trainer_config = self.trainer_config
        config_class: type[Wav2Vec2Config] = getattr(transformers, trainer_config.model_config_class)
        base = config_class()

        num_channels = trainer_config.x_num_channels if which == 'x' else trainer_config.y_num_channels

        conv_stride_base = trainer_config.model_conv_stride_base * num_channels
        conv_stride_other = trainer_config.model_conv_stride
        conv_kernel = base.conv_kernel.copy()
        conv_kernel[0] = conv_stride_base * trainer_config.conv_stride_multiplier
        model_width_scale = trainer_config.model_width_scale
        divisor = base.num_conv_pos_embedding_groups * trainer_config.num_attention_heads

        return config_class(
            conv_kernel=base.conv_kernel,
            conv_dim=[int(_ * model_width_scale) for _ in base.conv_dim],
            conv_stride=[conv_stride_base] + [conv_stride_other] * len(base.conv_stride[1:]),
            hidden_size=max(int(base.hidden_size * model_width_scale) // divisor, 1) * divisor,
            intermediate_size=max(int(base.intermediate_size * model_width_scale) // divisor, 1) * divisor,
            num_hidden_layers=trainer_config.num_hidden_layers,
            num_attention_heads=trainer_config.num_attention_heads,
            mask_time_length=base.mask_time_length,
        )

    def forward(self, **kwargs):
        return self.forward_data_batch(DataBatch(**kwargs)).model_dump()

    def forward_data_batch(self, data_batch: 'DataBatch'):
        x_emb = self.encoder_forward_pass(self.encoder_x, data_batch.x, attention_mask=data_batch.x_attention_mask)
        y_emb = self.encoder_forward_pass(self.encoder_y, data_batch.y, attention_mask=data_batch.y_attention_mask)
        similarity = self.similarity(x_emb, y_emb)
        loss = self._loss(similarity=similarity, target=data_batch.target)

        return ModelPrediction(
            x_emb=x_emb,
            y_emb=y_emb,
            loss=loss,
            similarity=similarity,
        )

    def encoder_forward_pass(self, encoder: PreTrainedModel, batch: torch.Tensor,
                             attention_mask: torch.LongTensor = None):
        return encoder(batch.type(DTYPE), attention_mask=attention_mask).last_hidden_state[:, 0]

    def _loss(self, similarity: torch.Tensor, target: torch.Tensor):
        logit_scale = torch.clamp(self.logit_scale, min=0, max=self.trainer_config.logit_scale_max)
        similarity_logits = similarity * logit_scale.exp()
        ce = nn.functional.cross_entropy
        return (
                ce(similarity_logits, target) +
                ce(similarity_logits.t(), target.t())
        ) / 2.0

    def similarity(self, x, y):
        if self.trainer_config.normalize:
            x = x / x.norm(dim=-1, keepdim=True)
            y = y / y.norm(dim=-1, keepdim=True)
        return x @ y.T

    def collate_same_size(self, items: Iterable[dict], which: str):
        vectors = [item[which] for item in items]
        if which == 'x':
            batch = np.stack(vectors)
        elif which == 'y':
            batch = np.stack([np.asarray(_).T.flatten() for _ in vectors])
        else:
            raise ValueError(which)
        return torch.tensor(batch)

    @staticmethod
    def metrics_from_similarity(similarity: torch.Tensor):
        max_score, predicted_indices = torch.max(similarity, dim=1)
        true_indices = torch.arange(len(similarity)).to(similarity.device)
        num_correct = torch.as_tensor(predicted_indices == true_indices).sum().item()
        num_total = len(true_indices)

        return Metrics(
            num_correct=num_correct,
            num_total=num_total,
            accuracy=num_correct / num_total,
        )


class HFModelTrainer(transformers.Trainer):
    train_dataset: TrainingDataset
    model: HFModel

    def __init__(self, train_dataset: TrainingDataset, **config):
        self.config = TrainingTasks.Config(**config)

        super().__init__(
            model=HFModel(**self.config.model_dump()),
            args=TrainingArguments(
                max_steps=self.config.max_steps,
                eval_steps=self.config.eval_steps,
                warmup_steps=self.config.warmup_steps,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                report_to=['wandb'] if self.config.wandb_enabled else [],

                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,

                learning_rate=self.config.learning_rate,
                optim=OptimizerNames.ADAMW_TORCH,
                lr_scheduler_type=SchedulerType.LINEAR,

                log_level=self.config.log_level,
                remove_unused_columns=False,
                evaluation_strategy=IntervalStrategy.STEPS,
                logging_strategy=IntervalStrategy.STEPS,
                save_strategy=IntervalStrategy.STEPS,
                label_names=['x', 'y', 'affinity'],

                use_cpu=self.config.use_cpu,
                use_mps_device=False,
                load_best_model_at_end=True,
                metric_for_best_model=self.config.metric_for_early_stopping,
                greater_is_better=True,
                save_safetensors=True,
                output_dir=Path(self.config.out_dir, wandb.run.id if self.config.wandb_enabled else '').as_posix(),
                dataloader_num_workers=self.config.dataloader_num_workers,
                dataloader_drop_last=True,
            ),
            train_dataset=train_dataset,
            eval_dataset=train_dataset.crops_eval_dataset,
            data_collator=self.data_batch_from_items,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)],
        )

    def data_batch_from_items(self, items: list[dict]):
        return DataBatch(
            x=self.model.collate_same_size(items, which='x'),
            y=self.model.collate_same_size(items, which='y'),
            target=torch.eye(len(items))
        ).to(dtype=DTYPE).model_dump()

    def evaluation_loop(self, dataloader: DataLoader, metric_key_prefix: str = "eval", **kwargs) -> EvalLoopOutput:
        self.model.eval()
        metrics_dict = self._compute_metrics(dataloader)
        metrics = pd.Series(metrics_dict).add_prefix(f"{metric_key_prefix}_").to_dict()
        # noinspection PyTypeChecker
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=metrics_dict['num_total']
        )

    def _compute_metrics(self, batches: Iterable[dict]):
        batches = itertools.islice(batches, self.config.limit_eval_batches)
        batches = tqdm(batches, desc='eval', leave=False, total=self.config.limit_eval_batches)

        with torch.no_grad():
            df = pd.DataFrame(self.model(**batch) for batch in batches)
            x_emb = torch.cat(df['x_emb'].tolist(), dim=0)
            y_emb = torch.cat(df['y_emb'].tolist(), dim=0)
            similarity = self.model.similarity(x_emb, y_emb)

        return TrainingMetrics(
            **HFModel.metrics_from_similarity(similarity).model_dump(),
            loss=df['loss'].mean(),
            logit_scale=self.model.logit_scale.item(),
        ).model_dump()


class ModelPrediction(PydanticClass):
    x_emb: TorchTensor
    y_emb: TorchTensor
    loss: TorchTensor
    similarity: TorchTensor


class DataBatch(PydanticClass):
    x: TorchTensor
    y: TorchTensor
    target: TorchTensor
    x_attention_mask: Optional[TorchTensor] = None
    y_attention_mask: Optional[TorchTensor] = None

    def to(self, **kwargs):
        return DataBatch(
            x=self.x.to(**kwargs),
            y=self.y.to(**kwargs),
            target=self.target.to(**kwargs),
            x_attention_mask=self.x_attention_mask.to(**kwargs) if self.x_attention_mask is not None else None,
            y_attention_mask=self.y_attention_mask.to(**kwargs) if self.y_attention_mask is not None else None,
        )


class Metrics(PydanticClass):
    accuracy: float
    num_correct: int
    num_total: int


class TrainingMetrics(Metrics):
    loss: float
    logit_scale: float


class Training:
    def train(self, **kwargs):
        TrainingTasks(**kwargs).train()

    def agent(self, **kwargs):
        TrainingTasks(**kwargs).agent()

    def sweep(self, **kwargs):
        wandb_sweep(**kwargs)


if __name__ == '__main__':
    Fire(Training)
