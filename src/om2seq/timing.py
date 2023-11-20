from devtools import debug
from fire import Fire

from om2seq.benchmarks import Benchmark
from utils.dataset_tasks import BaseTask


class Timing(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.benchmark = Benchmark()
        self.benchmark.init_dataset()
        self.benchmark.init_om2seq()
        self.qry_len = 100000
        self.crops = self.benchmark.generate_crops(qry_len=self.qry_len, limit=1000)
        self.eval_om_seq = self.benchmark.eval_om2seq(crops=self.crops)

    def om2seq(self):
        for i in range(2):
            with debug.timer('inference'):
                inference = self.eval_om_seq.inference(queries=self.crops)

            with debug.timer('retrieval'):
                retrieved = self.eval_om_seq.retrieve(query_embeddings=inference)
        # mapping_speed = len(self.crops) * self.qry_len / time
        # time, = timer.summary()
        # print(f'OM2Seq mapping speed: {mapping_speed / 1e6:.2f} Mbp/s')


if __name__ == '__main__':
    Fire(Timing)
