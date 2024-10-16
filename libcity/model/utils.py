import torch
from torch_geometric.utils import degree, to_undirected
from torch_geometric.utils import sort_edge_index, degree, add_remaining_self_loops, remove_self_loops, get_laplacian, \
    to_undirected, to_dense_adj, to_networkx
from torch_geometric.datasets import KarateClub
from torch_scatter import scatter
import torch_sparse
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, SGConv, GINConv, SAGEConv, GraphConv
from torch_geometric.utils import dropout_adj, degree, to_undirected

def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]


def get_base_model(name: str):
    def gat_wrapper(in_channels, out_channels):
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            heads=4
        )

    def gin_wrapper(in_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ELU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        return GINConv(mlp)

    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }

    return base_models[name]


def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):
    w = (1 - w) * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
    x = x.clone()
    x[:, drop_mask] = 0.
    return x


def feature_drop_weights_dense(x, node_c):
    x = x.abs()
    w = x.t() @ node_c
    s = (w - w.min()) / (w.max() - w.min()) # (f, )
    return s


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 0.7):
    edge_weights = (1 - edge_weights) * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    edge_weights = torch.where(edge_weights < 0, torch.tensor(0.0), edge_weights)
    sel_mask = (1 - torch.bernoulli(edge_weights)).to(torch.bool)
    return edge_index[:, sel_mask]
