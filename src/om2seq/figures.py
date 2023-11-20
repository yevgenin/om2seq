import json
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullFormatter, PercentFormatter

from om2seq.benchmarks import *
from om2seq.benchmarks import QRY_LEN, EvalDeepOM, EvalOM2Seq, EvalCombined
from om2seq.env import ENV
from utils.plot_utils import plot_confidence_intervals
from utils.wandb_utils import WandbRunData


class BenchmarkFigureBuilder(PydanticClassInputs):
    wandb_run_id: str = 'cwhltjyq'

    def export(self):
        api = wandb.Api()
        table = api.artifact(name=f'run-{self.wandb_run_id}-metrics:v0')
        table = json.loads(Path(table.download(root=ENV.LOCAL_OUT_DIR), 'metrics.table.json').read_text())
        data = BenchmarkFigure(
            wandb_run_id=self.wandb_run_id,
            data=pd.DataFrame(table['data'], columns=table['columns']).to_dict(orient='list'),
        )
        self.base_path.mkdir(exist_ok=True, parents=True)
        self.data_file.write_text(data.model_dump_json())

    @property
    def base_path(self):
        return Path(ENV.FIGURES_DIR, self.wandb_run_id)

    def load(self):
        file = self.data_file
        if not file.exists():
            self.export()
        assert file.exists()
        return BenchmarkFigure.model_validate_json(file.read_text())

    @property
    def data_file(self):
        return self.base_path / 'data.json'


class BenchmarkFigure(BenchmarkFigureBuilder):
    data: dict

    def plot(self, bionano_mapping_speed=True):
        plt.figure()
        self.base_path.mkdir(exist_ok=True)
        df = pd.DataFrame(self.data)
        for color, cls_ in zip(
                ['C0', 'C1', 'C2'],
                [EvalDeepOM, EvalOM2Seq, EvalCombined]
        ):
            column = cls_.evaluation_name + '_accuracy'
            if column in df:
                plt.plot(df['qry_len'], df[column], alpha=.8, color=color, marker='.', ls='-',
                         label=cls_.evaluation_name)
                plot_confidence_intervals(x_values=df['qry_len'], counts=df[cls_.evaluation_name + '_num_correct'],
                                          totals=df[cls_.evaluation_name + '_num_total'], color=color)

        bionano_df = self.deepom_bionano_mapping_results()
        bionano_df = bionano_df[bionano_df['qry_len'].between(df['qry_len'].min(), df['qry_len'].max())]
        plt.plot(bionano_df['qry_len'], bionano_df['accuracy'], alpha=.8, color='C3', marker='.', ls='--',
                 label='Commercial software')
        plot_confidence_intervals(x_values=bionano_df['qry_len'], counts=bionano_df['num_correct'],
                                  totals=bionano_df['num_total'], color='C3')

        plot_vs_len_setup()
        plt.ylabel('Accuracy')
        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(.1))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        plt.legend()

        plt.savefig(self.base_path / 'accuracy.pdf')
        plt.savefig(self.base_path / 'accuracy.png')

        plt.figure()
        for color, cls_ in zip(
                ['C0', 'C1', 'C2'],
                [EvalDeepOM, EvalOM2Seq, EvalCombined]
        ):
            column = cls_.evaluation_name + '_time'
            if column in df:
                t = df[column]
                x = df['qry_len']
                speed = x * df[cls_.evaluation_name + '_num_total'] / t
                plt.plot(x, speed / 1e6, alpha=.8, color=color, marker='.', ls='-',
                         label=cls_.evaluation_name)
        if bionano_mapping_speed:
            rate_bionano = self.deepom_bionano_mapping_speed()
            # plt.axhline(rate_bionano['rate'] / 1e6, color='C3', ls='--', label='Commercial software')
            plt.plot(rate_bionano['qry_len'], rate_bionano['rate'] / 1e6,
                     color='C3', marker='.', ms=10, ls='',
                     label='Commercial software')
            plt.ylim(1e-1, None)
        plot_vs_len_setup()
        plt.legend()
        plt.ylabel('Mapping speed (Mbp/s)')
        plt.yscale('log')
        plt.savefig(self.base_path / 'rate.pdf')
        plt.savefig(self.base_path / 'rate.png')

    def deepom_bionano_mapping_speed(self):
        bionano_mapping_speed = pd.read_csv(Path(ENV.FIGURES_DIR) / 'DeepOM_Bionano_mapping_speed.csv').iloc[0]
        mapping_speed_bp_per_sec = (bionano_mapping_speed['len_bp'] * bionano_mapping_speed['num_total'] /
                                   bionano_mapping_speed['time'])
        return dict(
            qry_len=bionano_mapping_speed['len_bp'],
            rate=mapping_speed_bp_per_sec,
        )

    def deepom_bionano_mapping_results(self):
        df = pd.read_csv(Path(ENV.FIGURES_DIR) / 'DeepOM_Bionano_mapping_results.csv')
        groupby_length = df.groupby("len_bp")["correct"]
        num_correct = groupby_length.sum()
        num_total = groupby_length.count()
        return pd.DataFrame(dict(
            qry_len=num_correct.index,
            accuracy=num_correct / num_total,
            num_correct=num_correct,
            num_total=num_total,
        ))


def plot_vs_len_setup():
    plt.xlabel(QRY_LEN)
    plt.xscale('log')
    plt.grid(which='both', axis='both')
    plt.xticks(np.stack([30, 40, 50, 70, 100, 150, 200]) * 1000)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, y: int(x // 1000)))
    ax.xaxis.set_minor_locator(MultipleLocator(10000))
    ax.xaxis.set_minor_formatter(NullFormatter())


def save_paper_figure(name):
    plt.savefig(ENV.FIGURES_DIR / f'{name}.pdf', bbox_inches='tight')
    plt.savefig(ENV.FIGURES_DIR / f'{name}.png', bbox_inches='tight')


class Figures:
    def all(self):
        self.benchmark()
        self.loss()
        self.data()

    def benchmark(self):
        BenchmarkFigureBuilder().load().plot()

    def loss(self):
        WandbRunData(wandb_run_name=Benchmark.Config().model_id_wandb_run_name).plot_loss()
        save_paper_figure('loss')

    def data(self, index=0):
        tds = TrainingDataset(add_random_shift=False, aligned_limit=10000, qry_len=50000)
        crop = tds[index]
        plt.figure(figsize=(15, 1))
        self._plot_crop(crop, tds)
        name = 'data'
        save_paper_figure(name)
        plt.xlim(800, 1600)

    def _plot_crop(self, crop, tds):
        cp = CropPlot(**crop, references=tds.references)
        plt.axis('off')
        cp.plot_image(cp.image, y_extent=(1, 0))
        cp.plot_lims()
        cp.plot_crop((3, 2), pos=2)
        margin = -.05
        plt.text(margin, .8, '(a)', transform=plt.gca().transAxes, ha='center', va='center', fontsize=20)
        plt.text(margin, .2, '(b)', transform=plt.gca().transAxes, ha='center', va='center', fontsize=20)
        plt.legend(handles=[
            Line2D([], [], color='m', label='limits'),
            Line2D([], [], color='b', label='reference'),
        ], loc='center right', bbox_to_anchor=(1.2, 0.5), fontsize=15, framealpha=1)
        plt.tight_layout()

    def crops(self):
        tds = TrainingDataset(add_random_shift=False, aligned_limit=10000, qry_len=50000)
        for i in range(5):
            crop = tds[i]

            plt.figure(figsize=(50, 1))
            self._plot_crop(crop, tds)
            save_paper_figure(i)


if __name__ == '__main__':
    Fire(Figures)
