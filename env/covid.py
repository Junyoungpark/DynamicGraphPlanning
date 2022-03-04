import numpy as np
from numpy.random import randint, choice, rand
import networkx as nx

import gym
from gym.spaces import MultiDiscrete

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils.random import stochastic_blockmodel_graph
from torch_geometric.utils import to_dense_adj, to_networkx

from collections import namedtuple
from copy import copy, deepcopy
from typing import Optional
from enum import Enum, IntEnum

import pdb

class NodeState(IntEnum):
    S=0
    E=1
    I=2
    R=3
    
class Measure(IntEnum): 
    business_as_usual=0 
    mask_mandate=1
    stay_at_home_order=2


SEIRconfig = namedtuple("SEIRconfig", ['β', 'σ', 'η', 'ζ'], defaults=[0.6, 0.8, 0.1, .05])

def randomList(m, n):
    """
    Given two integers m, n this function retruns a list of intergers of length m that sum to n.
    """
    arr = [0] * m
    for i in range(n) :
        arr[randint(n) % m] += 1
    return arr

score = lambda m : .6 if m == Measure.business_as_usual else (.2 if m == Measure.mask_mandate else .05)
cost = lambda m : .0 if m == Measure.business_as_usual else (1. if m == Measure.mask_mandate else 3.0)
inter_score = lambda m1, m2 : max(score(m1), score(m2))

format_input = lambda x: F.one_hot(torch.Tensor(np.vectorize(int)(x)).to(torch.int64), 
                                   num_classes=len(NodeState)).to(torch.float32)

format_data = lambda x: (format_input(x.x).to(torch.float32),
                         x.edge_index,
                         torch.Tensor(x.edge_attr))

class CovidSEIR(gym.Env):
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

    def __init__(self, n: int, m: int):
        self.n_nodes = n
        self.n_communities = m
        self.config = SEIRconfig()
        
        self.action_space = MultiDiscrete([len(Measure)] * self.n_communities)

        self.state = None
        self.community_labels = None
        self.community_sizes = None
        
    def seir(self, s: NodeState):
        if s == NodeState.S:
            return NodeState.S
        elif s == NodeState.E:
            return NodeState.I if rand() < self.config.σ else NodeState.E
        elif s == NodeState.I:
            return NodeState.R if rand() < self.config.η else NodeState.I
        return NodeState.S if rand() < self.config.ζ else NodeState.R
            
    def update_edge_weight(self, a):
        for i, (u, v) in enumerate(self.state.edge_index.T.tolist()):
            self.state.edge_attr[i] = inter_score(a[self.community_labels[u]], a[self.community_labels[v]])

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        
        # apply community measures
        self.update_edge_weight(action)
        
        # who is exposed
        i_nodes = np.where(self.state.x.squeeze() == NodeState.I)[0].tolist()
        s_nodes = np.where(self.state.x.squeeze() == NodeState.S)[0].tolist()
        i_node_mask = np.array([x in i_nodes for x in self.state.edge_index[1].tolist()])
        for v in s_nodes:
            # prob of catching it (given n infected neighbors)
            # = 1 - prob of not catching it from any of them
            # = 1 - ∏^n (1 - p^transmittion_n)
            mask = i_node_mask & (self.state.edge_index[0] == v).numpy()
            if any(mask) & (rand() > np.prod(self.state.edge_attr[mask])):
                self.state.x[v] = NodeState.I
                
        # transition graph temporally
        self.state.x = np.vectorize(self.seir)(self.state.x)
        
        done = all(self.state.x != NodeState.I)
        reward = np.sum(np.vectorize(cost)(action)*self.community_sizes)
        
        return deepcopy(self.state), reward, done, {}

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        if not seed == None:
            super().reset(seed=seed)
        
        probs = np.random.rand(self.n_communities, self.n_communities)
        probs = np.maximum((probs.T + probs)/4, 0.9*np.eye(self.n_communities))
        self.community_sizes = randomList(self.n_communities, self.n_nodes)
        edge_index = stochastic_blockmodel_graph(self.community_sizes, probs)
        x = choice(list(NodeState), size=self.n_nodes)
        
        self.state = Data(x=x, edge_index=edge_index, edge_attr=np.ones(edge_index.shape[1])*0.6)
        # following the defintion of stochastic block model generation (https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/random.html#stochastic_blockmodel_graph)
        self.community_labels = sum([[i]*b for i, b in enumerate(self.community_sizes)], [])
        
        return deepcopy(self.state)

    def render(self):
        g = torch_geometric.utils.to_networkx(data, to_undirected=True)
        nx.draw(g)
        
    def seed(self, n: int):
        super().reset(seed=seed)