import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap



    
    
    

import seaborn as sns
from tqdm.notebook import tqdm

from math import ceil
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import dense_diff_pool
from torch_geometric.loader import DenseDataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.nn.dense.dense_gcn_conv import DenseGCNConv
from torch_geometric.nn import DenseSAGEConv

from torch_geometric.utils.to_dense_adj import to_dense_adj
from torch_geometric.utils.to_dense_batch import to_dense_batch

torch.Tensor(0)

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=True, lin=True, track_running_stats=True):
        super().__init__()
        
        self.conv1 = DenseGCNConv(in_channels, hidden_channels, normalize)
        self.bn1 = nn.BatchNorm1d(hidden_channels, track_running_stats=track_running_stats)
        
        self.conv2 = DenseGCNConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = nn.BatchNorm1d(hidden_channels, track_running_stats=track_running_stats)
        
        #self.conv3 = DenseGCNConv(hidden_channels, hidden_channels, normalize)
        #self.bn3 = nn.BatchNorm1d(hidden_channels, track_running_stats=track_running_stats)
        
        self.conv4 = DenseGCNConv(hidden_channels, out_channels, normalize)
        self.bn4 = nn.BatchNorm1d(out_channels, track_running_stats=track_running_stats)
        
        if lin is True:
            self.lin = nn.Linear(2*hidden_channels + out_channels, out_channels)
        else:
            self.lin = None
            
    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        #x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))
        x4 = self.bn(4, F.relu(self.conv4(x2, adj, mask)))
        
        #x = torch.cat([x1, x2, x3, x4], dim=-1)
        x = torch.cat([x1, x2, x4], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x

            
class DiffPoolNet(torch.nn.Module):
    def __init__(self, max_nodes, num_features, num_hidden=64, out_channels=60, track_running_stats=True):
        super().__init__()

        num_nodes = ceil(0.3 * max_nodes)
        self.gnn1_pool = GNN(num_features, num_hidden, num_nodes, track_running_stats=track_running_stats)
        self.gnn1_embed = GNN(num_features, num_hidden, num_hidden, lin=False, track_running_stats=track_running_stats)

        num_nodes = ceil(0.3 * num_nodes)
        self.gnn2_pool = GNN(3 * num_hidden, num_hidden, num_nodes, track_running_stats=track_running_stats)
        self.gnn2_embed = GNN(3 * num_hidden, num_hidden, num_hidden, lin=False, track_running_stats=track_running_stats)
        
        num_nodes = ceil(0.3 * num_nodes)
        self.gnn3_pool = GNN(3 * num_hidden, num_hidden, num_nodes, track_running_stats=track_running_stats)
        self.gnn3_embed = GNN(3 * num_hidden, num_hidden, num_hidden, lin=False, track_running_stats=track_running_stats)
        

        self.gnn4_embed = GNN(3 * num_hidden, num_hidden, num_hidden, lin=False, track_running_stats=track_running_stats)

        self.lin1 = nn.Linear(3 * num_hidden, num_hidden)        
        self.lin2 = nn.Linear(num_hidden, out_channels)
        
        #self.lin3 = nn.Linear(2000, 640)
        #self.lin4 = nn.Linear(640, 64)
        #self.lin5 = nn.Linear(64, 1)
        

    def forward(self, batch):
        
        
        adj = to_dense_adj(batch.edge_index, batch=batch.batch, edge_attr=batch.edge_weight, max_num_nodes=None)
        x, _ = to_dense_batch(batch.x, batch.batch)
        batch.edge_index, batch.edge_weight, batch.batch, batch.ptr = None, None, None, None
        
        
        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s)
         
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        s = self.gnn3_pool(x, adj)
        x = self.gnn3_embed(x, adj)
        x, adj, l3, e3 = dense_diff_pool(x, adj, s)
        
        x = self.gnn4_embed(x, adj)

        x = x.mean(dim=1)        
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
                                            
        return x#, l1 + l2 + l3, e1 + e2 + e3
