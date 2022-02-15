import numpy as np
from collections import deque, OrderedDict
from rlkit.data_management.replay_buffer import ReplayBuffer
from warnings import warn
# from typing import Deque
import pdb

grep = lambda q, x : list(map(q.__getitem__, x))

class anyReplayBuffer(ReplayBuffer):

    def __init__(self, max_replay_buffer_size, replace = True):
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = deque([], max_replay_buffer_size)
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = deque([], max_replay_buffer_size)
        self._actions = deque([], max_replay_buffer_size)
        self._rewards = deque([], max_replay_buffer_size)
        self._terminals =  deque([], max_replay_buffer_size)
        self._replace = replace

    def add_sample(self, observation, action, reward, next_observation, terminal, env_info, **kwargs):
        self._observations.appendleft(observation)
        self._actions.appendleft(action)
        self._rewards.appendleft(reward)
        self._terminals.appendleft(terminal)
        self._next_obs.appendleft(next_observation)

    def terminate_episode(self):
        pass

    def random_batch(self, batch_size):
        indices = np.random.choice(self.get_size(), size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warn('Replace was set to false, but is temporarily set to true \
            because batch size is larger than current size of replay.')
        
        batch = dict(
            observations = grep(self._observations, indices),
            actions = grep(self._actions, indices),
            rewards = grep(self._rewards, indices),
            terminals = grep(self._terminals, indices),
            next_observations = grep(self._next_obs, indices),
        )
        
        return batch
    
    def get_size(self):
        return len(self._rewards)
        
    def rebuild_env_info_dict(self, idx):
        return self.batch_env_info_dict(idx)

    def batch_env_info_dict(self, indices):
        return {}

    def num_steps_can_sample(self):
        return self.get_size()

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self.get_size())
        ])