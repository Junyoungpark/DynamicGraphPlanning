import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.data import Data
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv, global_mean_pool

class GraphGNNModel(nn.Module):
    
    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of output features (usually number of classes)
            dp_rate_linear - Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs - Additional arguments for the GNNModel object
        """
        super().__init__()
        self.GNN = Sequential('x, edge_index, edge_weight', 
                              [(GCNConv(c_in, c_hidden), 'x, edge_index, edge_weight -> x'),
                               ReLU(inplace=True),
                               (GCNConv(c_hidden, c_hidden), 'x, edge_index, edge_weight -> x'),
                               Linear(c_hidden, int(c_hidden/2))])
        
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(int(c_hidden/2), c_out))
        
        self.activation = nn.Softmax(dim=1)
        self._device = 'cpu'

    def forward(self, x, edge_index, edge_weight, batch_idx):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx - Index of batch element for each node
        """
        x = self.GNN(x.to(self._device), edge_index.to(self._device), edge_weight.to(self._device))
        x = global_mean_pool(x, torch.Tensor(batch_idx).to(torch.int64).to(self._device)) # Average pooling
        x = self.activation(self.head(x))
        return x
    
#     def forward(self, batch: map, batch_idx):
#         ff = lambda x: self.forward(*x, env.community_labels)
#         return map(ff, batch)
    
    def to(self, device):
        super().to(device)
        self._device = device