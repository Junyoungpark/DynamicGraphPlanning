import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
from numpy.random import rand

from rlkit.policies.base import Policy
# from env import format_data

class argmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qf, format_data, dim=1, args=[]):
        super().__init__()
        self.qf = qf
        self.args = args
        self.format_fn = format_data
        self.dim = dim

    def get_action(self, obs):
        q_values = self.qf(*self.format_fn(obs), *self.args)
        return q_values.cpu().detach().numpy().argmax(self.dim), {}

# redundant code - clean this up
class epsilonGreedyPolicy(nn.Module, Policy):
    def __init__(self, qf, format_data, space, eps=0.1, dim=1, args=[]):
        super().__init__()
        self.qf = qf
        self.args = args
        self.eps = eps
        self.aspace = space
        self.format_fn = format_data
        self.dim = dim

    def get_action(self, obs):
        if rand() < self.eps:
            return self.aspace.sample(), {}
        q_values = self.qf(*self.format_fn(obs), *self.args)
        return q_values.cpu().detach().numpy().argmax(self.dim), {}