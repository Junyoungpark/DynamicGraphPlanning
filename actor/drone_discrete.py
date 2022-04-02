from warnings import warn

import numpy as np
import torch
import torch.nn as nn
from rlkit.policies.base import Policy
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, SAGEConv

from env.drone_delivery import action


# ------------------ Helpers

def interleave_lists(list1, list2):
    out = list1 + list2
    out[::2] = list1
    out[1::2] = list2
    return out


# -------------- Heuristic policy (baseline)

def pick(x):
    probs = np.array([x[0] > 0, x[0] < 0, x[1] > 0, x[1] < 0, x[1] == x[0] == 0], dtype=int)
    return np.random.choice(action, p=probs / sum(probs))


class sysRolloutPolicy(nn.Module, Policy):
    def __init__(self, n_agents=-1, device='cpu'):
        super().__init__()
        if n_agents <= 0:
            assert n_agents != 0, "Yeah nah!! this must be a mistake, you don't have any agents in your scene"
            warn("Just double checking... You have " + str(-n_agents) + " goal regions?")

        self.n = n_agents
        self.device = device

    def get_action(self, obs):
        idx = torch.cdist(obs.x[:self.n, :-1], obs.x[self.n:, :-1], p=1).min(1).indices
        dis = (obs.x[idx + self.n, :-1] - obs.x[:self.n, :-1])
        return torch.Tensor([pick(d) for d in dis]).to(torch.long).to(self.device), {}


# -------------- GCN model

class droneDeliveryModel(nn.Module):

    def __init__(self, c_in, c_out, c_hidden=[], n_linear=1, bounds=None, **kwargs):

        super().__init__()

        assert kwargs['n_agents'] > 0, "Yeah nah!! this must be a mistake, you don't have any agents in your scene"
        assert kwargs['n_goals'] > 0, "Yeah nah!! this must be a mistake, you don't have any goal regions in your scene"

        activation_fn = kwargs['activation'] if 'activation' in kwargs.keys() else ReLU(inplace=True)

        assert len(c_hidden) > 0, "Hidden dimension can not be zero => no GCN layer."
        layer_size = [c_in] + c_hidden + [c_out]
        n_sage = len(layer_size) - n_linear - 1

        layers = [(SAGEConv(layer_size[i], layer_size[i + 1]), 'x, edge_index -> x')
                  if i < n_sage else
                  Linear(layer_size[i], layer_size[i + 1])
                  for i in range(len(c_hidden) + 1)]
        layers = interleave_lists(layers, [activation_fn] * (len(layer_size) - 2))

        self.model = Sequential('x, edge_index', layers)

        self._device = 'cpu'
        # self._upper_bound = bounds
        self.register_buffer('_upper_bound', bounds)
        self._n = kwargs['n_agents']  # drones
        self._g = kwargs['n_goals']  # goals
        self._a = c_out  # actions

    def forward(self, x):
        y = x.x
        if self._upper_bound is not None:
            y = y.div(self._upper_bound - 1)
        return self.model(y, x.edge_index).reshape(-1, self._n + self._g, self._a)[:, :self._n, :].reshape(-1, self._a)

    def to(self, device):
        super().to(device)
        self._device = device
        if self._upper_bound is not None:
            self._upper_bound = self._upper_bound.to(device)
