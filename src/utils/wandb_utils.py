import json
from pathlib import Path

import pandas as pd
import safetensors.torch
import wandb
from devtools import debug
from matplotlib import pyplot as plt

from om2seq.env import ENV


class WandbRunData:
    def __init__(self, wandb_run_name: str, artifact_type: str = 'checkpoint', artifact_version: str = 'latest'):
        debug(wandb_run_name)
        self.api = wandb.Api()
        self.target_path = (Path(ENV.LOCAL_OUT_DIR) / 'wandb-runs' / wandb_run_name).as_posix()
        self.artifact = self.api.artifact(name=f'ogm/ogm/{artifact_type}-{wandb_run_name}:{artifact_version}',
                                          type='model')
        self.artifact.download(self.target_path)
        self.run_config = dict(self.api.run(f'ogm/ogm/{wandb_run_name}').config)
        self.state_dict = safetensors.torch.load_file(Path(debug(self.target_path)) / 'model.safetensors')

        trainer_state_json_file = (Path(self.target_path) / 'trainer_state.json')
        if trainer_state_json_file.exists():
            log_history = json.loads(trainer_state_json_file.read_text())["log_history"]
            self.log_history = pd.DataFrame(log_history)

    def plot_loss(self, loss_step=100):
        df = self.log_history.set_index('step')
        print(df.keys())
        df['loss'].dropna()[::loss_step].plot(color='C0')
        df['eval_loss'].dropna().plot(color='C1', ax=plt.gca())
        # plt.yscale('log')
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.legend(['train', 'validation'])
        plt.grid(which='both')

    def plot_accuracy(self):
        df = self.log_history.dropna(subset='eval_accuracy')
        df.plot(x='step', y='eval_accuracy')
