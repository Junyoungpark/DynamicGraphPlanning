import numpy as np
from numpy.random import randint, choice, rand
import networkx as nx

import gym
from gym.spaces import MultiDiscrete

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils.random import erdos_renyi_graph
from torch_geometric.utils import to_dense_adj, to_networkx

from collections import namedtuple
from copy import copy, deepcopy
from typing import Optional
from enum import Enum, IntEnum

import pdb

class status(IntEnum):
    good=0
    faulty=1
    dead=2
    
class load(IntEnum):
    idle=0
    loaded=1
    success=2
    
class action(IntEnum): 
    noop=0 
    reboot=1
    
sysconfig = namedtuple("sysconfig", 
                       ['p_fail_base', 'p_fail_bonus', 'p_dead_base', 'p_dead_bonus',
                        'p_load', 'p_doneG', 'p_doneF',
                        'discount', 'reboot_penalty', 'job_done'], 
                       defaults=[.4,.2,.1,.5,.6,.9,.6,.9,-.25, 1.])

format_input = lambda x: F.one_hot(x, num_classes=len(status)).reshape(-1,len(status)+len(load)).to(torch.float32)

format_data = lambda x: (format_input(x.x), x.edge_index)

class sysAdmin(gym.Env):
    """
    ### Description
    
    ### Action Space
    Each community within the graph has governance of given nodes, an action notes the measure each 
    community is taking at any given time during simulation.
    
    ### State Space
    The state is defined as an arbitrary input graph where each node has an associate SEIR state. The 
    edge weights reflect transmission probabilities if either node is infectious.
    
    ### Rewards
    The reward model is defined as the sum of cost associated with each measure weighted by the size
    of the affected community.
    
    ### Starting State
    Randomly initilized stochastic Blockmodelgraph with predefined number of nodes and communities.
    
    ### Episode Termination
    The simulation terminates when no node is infectious anymore. At this point the the desease can
    no longer spread.
    
    ### Arguments
    No additional arguments are currently supported.
    """

    def __init__(self, nnodes: int, njobs: int):
        self.n_nodes = nnodes
        self.n_jobs = njobs
        self.config = sysconfig()
        self.topology = 'random'
        
        self.action_space = MultiDiscrete([len(action)] * self.n_nodes)
        self.state = None
        
    def reward(self, a):
        loads = self.state.x[:, 1]
        return torch.sum(loads==load.success) * self.config.job_done + torch.sum(a==action.reboot) * self.config.reboot_penalty
    
    def _step(self, s):
        if s[0] == status.dead:
            return load.idle
        elif s[1] == load.idle:
            getload = rand() < (self.config.p_load * (self.n_jobs > 0))
            self.n_jobs -= getload
            return load.loaded if getload else load.idle
        elif s[1] == load.success:
            return load.idle
        
        p_done = self.config.p_doneF if s[0] == status.faulty else self.config.p_doneG
        return load.success if rand() < p_done else load.loaded
                    
    def step(self, a):
        err_msg = f"{a!r} ({type(a)}) invalid"
        assert self.action_space.contains(a), err_msg
        
        self.count = self.count - (self.state.x[:, 0]!=status.dead).sum().item()
        done = self.count <= 0
        
        a = torch.Tensor(a)
        reward = self.reward(a)
                
        # who is exposed
        i_nodes = torch.where(self.state.x[:, 0] == status.faulty)[0]
        s_nodes = torch.where(self.state.x[:, 0] == status.good)[0]
        i_node_mask = torch.Tensor([x.item() in i_nodes for x in self.state.edge_index[1]]).to(torch.bool)
        for v in s_nodes:
            # prob of catching it (given n infected neighbors)
            # = 1 - prob of not catching it from any of them
            # = 1 - ∏^n (1 - p^transmittion_n)
            mask = i_node_mask & (self.state.edge_index[0] == v.item())
            p_fail = mask.sum().item() * self.config.p_fail_bonus + self.config.p_fail_base
            if rand() < p_fail:
                self.state.x[v, 0] = status.faulty
        for v in i_nodes:
            if rand() < self.config.p_dead_base:
                self.state.x[v, 0] = status.dead
        if any(a == action.reboot):
            self.state.x[:, 0] = torch.where(a == action.reboot,
                                             torch.zeros((self.n_nodes,), dtype=torch.int64),
                                             self.state.x[:, 0])
            self.state.x[:, 1] = torch.where(a == action.reboot,
                                             torch.zeros((self.n_nodes,), dtype=torch.int64),
                                             self.state.x[:, 1])
                
        # transition graph temporally
        self.state.x[:, 1] = torch.Tensor(np.apply_along_axis(self._step, -1, self.state.x.numpy()))    
        
        return deepcopy(self.state), deepcopy(reward.item()), deepcopy(done), {}

    def reset(self, seed: Optional[int] = None, topology: str = 'random'):
        if not seed == None:
            super().reset(seed=seed)
        self.topology = topology
        
        if self.topology == 'random':
            edge_index = erdos_renyi_graph(self.n_nodes, 0.75, directed=False)
        elif self.topology == 'star':
            arr = torch.arange(1, self.n_nodes)
            edge_index = torch.stack([arr, torch.zeros(self.n_nodes-1, dtype=torch.int64)])
            edge_index = torch.hstack([edge_index, edge_index.flip(0)])
        elif self.topology == 'ring':
            arr = torch.arange(self.n_nodes)
            edge_index = torch.vstack([arr, arr.roll(-1,0)])
            edge_index = torch.hstack([edge_index, edge_index.flip(0)])
        else:
            err_msg = f"Unknown topology. Choose among 'ring', 'star', or 'random'."
            assert False, err_msg
            
        x = torch.zeros((self.n_nodes, 2), dtype=torch.int64) #torch.randint(high=len(status), size=(3,2))
        self.state = Data(x=x, edge_index=edge_index)
        
        return deepcopy(self.state)

    def render(self):
        g = torch_geometric.utils.to_networkx(self.state, to_undirected=True)
        colors = np.array(['green', 'blue', 'red'])
        color_map = colors[self.state.x.numpy()[:, 0]]
        labeldict = {i: 'L' if v==load.loaded else ('I' if v==load.idle else 'S')  for i, v in enumerate(self.state.x[:, 1])}
        nx.draw(g, node_color=color_map, labels=labeldict)
        
    def seed(self, n: int):
        super().reset(seed=seed)