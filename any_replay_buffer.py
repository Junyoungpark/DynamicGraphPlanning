import numpy as np
import pdb

from collections import deque, OrderedDict
from rlkit.data_management.replay_buffer import ReplayBuffer
from torch import is_tensor
from warnings import warn

grep = lambda q, x : list(map(q.__getitem__, x))
softmax = lambda x : np.exp(x)/sum(np.exp(x))

class anyReplayBuffer(ReplayBuffer):

    def __init__(self, max_replay_buffer_size, replace = True, prioritized=False):
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
        self._prioritized = prioritized
        self._weights = deque([], max_replay_buffer_size)
            

    def add_sample(self, observation, action, reward, next_observation, terminal, env_info, **kwargs):
        self._observations.appendleft(observation)
        self._actions.appendleft(action)
        self._rewards.appendleft(reward)
        self._terminals.appendleft(terminal)
        self._next_obs.appendleft(next_observation)
        self._weights.appendleft(reward.sum().item() if is_tensor(reward) else reward)

    def terminate_episode(self):
        pass

    def random_batch(self, batch_size):
#         pdb.set_trace()
        prio = softmax(self._weights) if self._prioritized else None
        indices = np.random.choice(self.get_size(), 
                                   size=batch_size, p=prio, 
                                   replace=self._replace or self._size < batch_size)
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