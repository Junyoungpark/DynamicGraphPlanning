import numpy as np
import torch
import torch.nn as nn

from numpy.random import rand
from rlkit.policies.base import Policy
    
class argmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qf, dim=1):
        super().__init__()
        self.qf = qf
        self.dim = dim

    def get_action(self, obs):
        q_values = self.qf(obs)
        return q_values.cpu().detach().numpy().argmax(self.dim), {}

# redundant code - clean this up
class epsilonGreedyPolicy(nn.Module, Policy):
    def __init__(self, qf, space, eps=0.1, dim=1, sim_annealing_fac=1.0):
        super().__init__()
        self.qf = qf
        self.aspace = space
        
        self.eps = eps
        self.dim = dim
        self.saf = np.clip(sim_annealing_fac, .0, 1.)
        
    def simulated_annealing(self):
        self.eps *= self.saf

    def get_action(self, obs):
        if rand() < self.eps:
            return self.aspace.sample(), {}
        q_values = self.qf(obs)
        return q_values.cpu().detach().numpy().argmax(self.dim), {}
    

#### Old version
# class argmaxDiscretePolicy(nn.Module, Policy):
#     def __init__(self, qf, format_data, dim=1, args=[]):
#         super().__init__()
#         self.qf = qf
#         self.args = args
#         self.format_fn = format_data
#         self.dim = dim

#     def get_action(self, obs):
#         q_values = self.qf(*self.format_fn(obs), *self.args)
#         return q_values.cpu().detach().numpy().argmax(self.dim), {}

# # redundant code - clean this up
# class epsilonGreedyPolicy(nn.Module, Policy):
#     def __init__(self, qf, format_data, space, eps=0.1, dim=1, args=[]):
#         super().__init__()
#         self.qf = qf
#         self.args = args
#         self.eps = eps
#         self.aspace = space
#         self.format_fn = format_data
#         self.dim = dim

#     def get_action(self, obs):
#         if rand() < self.eps:
#             return self.aspace.sample(), {}
#         q_values = self.qf(*self.format_fn(obs), *self.args)
#         return q_values.cpu().detach().numpy().argmax(self.dim), {}
    