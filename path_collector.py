import ray
import multiprocessing as mp
from collections import deque, OrderedDict

from rlkit.samplers.data_collector.base import PathCollector

from rollout_functions import rollout


@ray.remote(num_cpus=1)
def remote_rollout():
    pass


class MdpPathCollector(PathCollector):
    def __init__(self, env, policy, max_num_epoch_paths_saved=None, rollout_fn=rollout,
                 parallelize=False, parallelize_method: str = 'mp'):
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

        self._rollout_fn = rollout_fn
        self._multithreading = parallelize
        self._parallelize_method = parallelize_method

    def collect_new_paths(self, n_paths, max_path_length, discard_incomplete_paths=False, flatten=False):
        paths = []
        if self._multithreading:
            if self._parallelize_method == 'mp':
                pool = mp.Pool(mp.cpu_count())
                paths = pool.starmap(self._rollout_fn, [(self._env, self._policy, max_path_length)] * n_paths)
                pool.close()
            else:
                # ray-based parallelization
                # NOTE: this would be very suboptimal implementation.
                pass
        else:
            for _ in range(n_paths):
                path = self._rollout_fn(self._env, self._policy, max_path_length=max_path_length)
                paths.append(path)
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', sum(path_lens)),
            ('number of epoch paths', len(self._epoch_paths)),
        ])
        return stats
