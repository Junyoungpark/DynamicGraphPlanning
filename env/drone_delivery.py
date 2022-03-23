import gym
import networkx as nx
import numpy as np
import pdb
import torch
import torch_geometric

from collections import namedtuple
from copy import copy, deepcopy
from enum import Enum, IntEnum
from gym.spaces import Box, MultiDiscrete
from numpy.random import choice, randint, rand, uniform
from typing import Optional

from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_networkx, to_undirected

sysconfig = namedtuple("sysconfig", 
                       ['maxX', 'maxY', 'goal_reward', 'collision_penalty', 'distance_penalty'], 
                       defaults=[4, 4, 1., -.1, -.01])

actions = namedtuple("actions", 
                    ['right', 'left', 'up', 'down', 'noop'], 
                    defaults=[np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)])
action = actions()

def fully_connect_graph(n_nodes):
    """ Connect the graph s.t. all drones and goals are interconnected. """
    
    idx = torch.combinations(torch.arange(n_nodes), r=2)
    return to_undirected(idx.t(), num_nodes=n_nodes)

def connect_graph(n_drones, n_goal, reverse=False):
    """ Connect the graph s.t. all drones are interconnected and each goal connects to all drones. """
    
    idx = torch.combinations(torch.arange(n_drones), r=2)
    edge_index = to_undirected(idx.t(), num_nodes=n_drones)

    arr = torch.arange(n_drones, n_drones+n_goal)
    for i in range(n_drones):
        goal2drone = torch.stack([arr, i*torch.ones(n_goal, dtype=torch.int64)]) if not reverse else \
                     torch.stack([i*torch.ones(n_goal, dtype=torch.int64), arr])
        edge_index = torch.hstack((edge_index, goal2drone))
    return edge_index

def sparse_connect_graph(n_drones, n_goal, reverse=False, undirected=False):
    """ Connect the graph s.t. all drones are connected to each goal only. Requires a model with GCN depth ≥ 2 """
    
    arr = torch.arange(n_drones, n_drones+n_goal)
    edge_index = torch.hstack([torch.stack([arr, i*torch.ones(n_goal, dtype=torch.int64)]) \
                               for i in range(n_drones)])
    if undirected:
        return to_undirected(edge_index)
    if reverse:
        return edge_index.flip(0)
    return edge_index

class droneDelivery(gym.Env):
    """
    ### Description
    
    ### Action Space
    Each agent in the scene can move horizontaly (R or L), vertically (U or D) or not at all.
    
    ### State Space
    The state is defined as an arbitrary input array of positions of n drones, appended by the location
    of the goal region.
    
    ### Rewards
    The reward of +1 is given to the system for when all drones reached a valid goal region, and a penality for any two drones
    colliding. Since this describes a discrete environment, to agents are considered in collision iif they occupy the same cell.
    
    ### Starting State
    Randomly initilized input array.
    
    ### Episode Termination
    When all drones have reached the goal region.
    
    ### Arguments
    No additional arguments are currently supported.
    """

    def __init__(self, args, device='cpu'):
        self.config = sysconfig(maxX=args.maze_size, maxY=args.maze_size)
        self.ndrones = args.ndrones
        self.ngoals = args.ngoals
        
        self.aspace = MultiDiscrete([len(action)]*self.ndrones)
        self.a2vecmap = torch.Tensor([[1., 0.],
                                      [-1, 0.],
                                      [0., 1.],
                                      [0, -1.],
                                      [0., 0.]]).to(device)
        self.sspace = MultiDiscrete([self.config.maxX, self.config.maxY, 2])
        self.state = None
        
        if args.graph_type == "full":
            self.generate_g = lambda : fully_connect_graph(self.ndrones+self.ngoals) 
        elif args.graph_type == "sparse":
            self.generate_g = lambda : sparse_connect_graph(self.ndrones, self.ngoals, undirected=True) 
        elif args.graph_type == "light":
            self.generate_g = lambda : connect_graph(self.ndrones, self.ngoals) 
        else:
            assert False, "Unspecified graph type."
        
        self._device = device
        
    def get_distances(self):
        dis = torch.cdist(self.state.x[:self.ndrones, :-1], self.state.x[-self.ngoals:, :-1], p=1)
        return dis.min(1).values
    
    def in_collision(self):
        dis = torch.cdist(self.state.x[:self.ndrones, :-1], self.state.x[:self.ndrones, :-1], p=1)
        return torch.triu((dis == 0).float(), 1).sum().item()
    
    def is_terminal(self):
        return (self.get_distances() == 0).all()
    
    def get_size(self):
        return torch.Tensor([self.config.maxX, self.config.maxY, 2])
    
    def get_channels(self):
        """ Returns in, out channel sizes per agent. """
        return self.sspace.shape[0], self.aspace.nvec[0]
            
    def get_max_len(self):
        return self.config.maxX + self.config.maxY - 1
        
    def reward(self, a):
        done = self.is_terminal().float().item()
        return self.config.goal_reward * done + \
               self.config.collision_penalty * self.in_collision() * (1 - done) + \
               self.config.distance_penalty * self.get_distances() #.sum().item()
                        
    def step(self, a):
        err_msg = f"{a!r} ({type(a)}) is not a valid action."
        assert self.aspace.contains(a), err_msg
        
        reward = self.reward(a)
        a = self.a2vecmap[a]
        done = self.is_terminal().item()   
    
        self.state.x[:self.ndrones, :-1] = (self.state.x[:self.ndrones, :-1]+a).clamp(min=0, max=self.config.maxX)
        
        return deepcopy(self.state), deepcopy(reward), deepcopy(done), {}

    def reset(self, seed: Optional[int] = None):
        if not seed == None:
            super().reset(seed=seed)
        
        x = torch.Tensor(np.stack([self.sspace.sample() for _ in range(self.ndrones+self.ngoals)]))
        # reset the state flags: +1 agent, -1 goal
        x[:, -1] = 1
        x[self.ndrones:, -1] = -1        
        
        self.state = Data(x=x, edge_index=self.generate_g()).to(self._device)
        
        return deepcopy(self.state)

    def render(self, s = None):
        if not s:
            s = self.state
        g = torch_geometric.utils.to_networkx(s, to_undirected=False)
        colors = np.array(['green']*self.ndrones+['yellow']*self.ngoals)
        pos = {i: x[:2].numpy() for i, x in enumerate(self.state.x)}
        nx.draw(g, pos=pos, node_color=colors)
    
    def seed(self, n: int):
        super().seed(n)
        self.aspace.seed(n)
        self.sspace.seed(n)
        
    def to(self, device):
        self._device = device
        self.a2vecmap = self.a2vecmap.to(device)
        if self.state:
            self.state = self.state.to(device)
            
            