import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from env.drone_delivery import action
from rlkit.policies.base import Policy
from torch.nn import Linear, ReLU, Softmax
from torch_geometric.nn import Sequential, GCNConv, SAGEConv
from warnings import warn

#-------------- Heuristic policy (baseline)

def pick(x):
    probs = np.array([x[0]>0, x[0]<0, x[1]>0, x[1]<0, x[1]==x[0]==0], dtype=int)
    return np.random.choice(action, p=probs/sum(probs))

class sysRolloutPolicy(nn.Module, Policy):
    def __init__(self, n_agents=-1):
        super().__init__()
        if n_agents <= 0:
            assert n_agents != 0, "Yeah nah!! this must be a mistake, you don't have any agents in your scene"
            warn("Just double checking... You have "+str(-n_agents)+" goal regions?")
            
        self.n = n_agents

    def get_action(self, obs):
        idx = torch.cdist(obs.x[:self.n, :-1], obs.x[self.n:, :-1], p=1).min(1).indices
        dis = (obs.x[idx+self.n, :-1] - obs.x[:self.n, :-1])
        return [pick(d) for d in dis], {}
    
#-------------- GCN model   
        
class droneDeliveryModel(nn.Module):
    
    def __init__(self, c_in, c_out, c_hidden=32, n_agents=-1, bounds=None, **kwargs):
        
        super().__init__()
        
        if n_agents <= 0:
            assert n_agents != 0, "Yeah nah!! this must be a mistake, you don't have any agents in your scene"
            warn("Just double checking... You have "+str(-n_agents)+" goal regions?")
        
        self.model = Sequential('x, edge_index', [
            (SAGEConv(c_in, c_hidden), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (SAGEConv(c_hidden, c_hidden), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (SAGEConv(c_hidden, c_hidden), 'x, edge_index -> x'),
            ReLU(inplace=True),
            Linear(c_hidden, c_out),
#             nn.Softmax(dim=-1) # no freaking softmax
        ])
        
        self._device = 'cpu'
        self._upper_bound = bounds
        self._n = n_agents

    def forward(self, x):
        y = x.x
        if self._upper_bound is not None:
            y = y.div(self._upper_bound-1)
        return self.model(y, x.edge_index)[:self._n]
    
    def to(self, device):
        super().to(device)
        self._device = device
        if self._upper_bound is not None:
            self._upper_bound = self._upper_bound.to(device)
            
            