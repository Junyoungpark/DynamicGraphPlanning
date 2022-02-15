import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
from numpy.random import rand

from rlkit.policies.base import Policy
from env import format_data

class argmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qf, args=None):
        super().__init__()
        self.qf = qf
        self.args = args

    def get_action(self, obs):
        q_values = self.qf(*format_data(obs), self.args)
        return q_values.cpu().detach().numpy().argmax(1), {}

# redundant code - clean this up
class epsilonGreedyPolicy(nn.Module, Policy):
    def __init__(self, qf, space, eps=0.05, args=None):
        super().__init__()
        self.qf = qf
        self.args = args
#         self.greedy = ArgmaxDiscretePolicy(qf, args)
        self.eps = eps
        self.aspace = space

    def get_action(self, obs):
        if rand() < self.eps:
            return self.aspace.sample(), {}
#         return self.greedy(obs)
        q_values = self.qf(*format_data(obs), self.args)
        return q_values.cpu().detach().numpy().argmax(1), {}