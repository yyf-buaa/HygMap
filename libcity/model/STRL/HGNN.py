import math
import torch
import copy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.sparse import coo_matrix

from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter, scatter_softmax
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional, Callable

from sklearn.metrics import average_precision_score as apscore  # True / Pred
from sklearn.metrics import roc_auc_score as auroc

from torch_geometric.nn.conv import MessagePassing


class GeoConv(MessagePassing):

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        degree_matrix = np.load(
            './raw_data/{}/degree_{}.npy'.format(self.dataset, self.dataset)).astype(np.int64)
        self.degree_matrix = torch.from_numpy(degree_matrix).to(self.device)

        dist_matrix = np.load(
            './raw_data/{}/dist_{}.npy'.format(self.dataset, self.dataset)).astype(np.int64)
        self.dist_matrix = torch.from_numpy(dist_matrix).to(self.device)
        self.W_node = nn.Linear(in_features, out_features, bias=False)
        self.W_edge = nn.Linear(in_features, out_features, bias=False)
        self.attention = nn.Linear(2*in_features,1)
        self.w_dist = nn.Linear(in_features, 1)
        self.w_degree = nn.Linear(in_features, 1)



    def forward(self, X, E, hyperedge_index, alpha, beta, X0, E0, degV, degE):
        num_nodes = X.shape[0]
        num_edges = int(hyperedge_index[1][-1]) + 1

        De = scatter_add(X.new_ones(hyperedge_index.shape[1]),
                         hyperedge_index[1], dim=0, dim_size=num_edges)

        De_inv = 1.0 / De
        De_inv[De_inv == float('inf')] = 0

        norm_n2e = De_inv[hyperedge_index[1]]

        N = X.shape[0]
        attention_weights_v2e = self.compute_attention(E, X, hyperedge_index).unsqueeze(1)
        attention_weights_e2v = self.compute_attention(X, E, hyperedge_index.flip([0])).unsqueeze(1)
        # import ipdb
        # ipdb.set_trace()
        #wave 1
        Xe1 = scatter((X[hyperedge_index[0]] * attention_weights_v2e), hyperedge_index[1], dim=0, reduce='sum')
        Xe1 = Xe1 * degE
        Xv1 = scatter((Xe1[hyperedge_index[1]] * attention_weights_e2v),hyperedge_index[0],dim=0,reduce='sum')

        Xv1 = Xv1 * degV
        # Xi = (1 - alpha) * X + alpha * X0
        # X = (1 - beta) * Xi + beta * self.W(Xi)

        #wave 2
        Xv2 = scatter((E[hyperedge_index[1]] * attention_weights_e2v),hyperedge_index[0],dim=0,reduce='sum')

        Xv2 = Xv2 * degV
        Xe2 = scatter((Xv2[hyperedge_index[0]] * attention_weights_v2e), hyperedge_index[1], dim=0, reduce='sum')
        Xe2 = Xe2 * degE

        X = (Xv1+Xv2)/2
        Xe = (Xe1+Xe2)/2
        Xi = (1 - alpha) * X + alpha * X0
        X = (1 - beta) * Xi + beta * self.W_node(Xi)

        Xei = (1 - alpha) * Xe + alpha * E0
        Xe= (1 - beta) * Xei + beta * self.W_edge(Xei)
        return X, Xe

    def reset_parameters(self):
        self.W_node.reset_parameters()
        self.W_edge.reset_parameters()

    def compute_attention(self, features_1, features_2, hyperedge_index):
        row, col = hyperedge_index
        concat_features = torch.cat([features_2[row], features_1[col]], dim=1)
        attention_scores = self.attention(concat_features).squeeze(-1)
        dist_attn_scores = self.w_dist(self.dist_matrix[row,col])
        angle_attn_scores = self.w_degree(self.degree_matrix[row,col])
        attention_weights = scatter_softmax(attention_scores+dist_attn_scores+angle_attn_scores, col)
        return attention_weights
def get_degree_of_hypergraph(hyperedge_index, device) : ## For UNIGCNII

    ones = torch.ones(hyperedge_index[0].shape[0], dtype = torch.int64).to(device)
    dV = scatter(src = ones,
        index = hyperedge_index[0], reduce = 'sum')

    dE = scatter(src = dV[hyperedge_index[0]],
                index= hyperedge_index[1], reduce = 'mean')

    dV = dV.pow(-0.5)
    dE = dE.pow(-0.5)
    dV[dV.isinf()] = 1
    dE[dE.isinf()] = 1

    del ones

    return dV.reshape(-1, 1), dE.reshape(-1, 1)


class HyperEncoder(nn.Module):

    def __init__(self, in_dim, edge_dim, node_dim, drop_p=0.2, num_layers=4, cached=False):
        super(HyperEncoder, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = torch.nn.ReLU()
        self.DropLayer = torch.nn.Dropout(p=drop_p)
        self.convs = nn.ModuleList()

        self.lamda, self.alpha = 0.5, 0.1
        self.convs.append(torch.nn.Linear(self.in_dim, self.node_dim))
        for _ in range(self.num_layers):
            self.convs.append(GeoConv(self.node_dim, self.node_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, e: Tensor , hyperedge_index: Tensor):

        degV, degE = get_degree_of_hypergraph(hyperedge_index, hyperedge_index.device)  # Getting degree
        degV = degV.reshape(-1, 1)
        degE = degE.reshape(-1, 1)

        x = self.DropLayer(x)
        x = torch.relu(self.convs[0](x))
        x0 = x
        e0 = e
        lamda, alpha = 0.5, 0.1
        for i, conv in enumerate(self.convs[1:]):
            x = self.DropLayer(x)
            e = self.DropLayer(e)
            beta = math.log(lamda / (i + 1) + 1)
            if i == len(self.convs[1:]) - 1:
                x,e = conv(x,e, hyperedge_index, alpha, beta, x0, e0, degV, degE)
            else:
                x,e = conv(x,e, hyperedge_index, alpha, beta, x0, e0, degV, degE)
                x = torch.relu(x)
                e = torch.relu(e)
        return x,e  # Only Returns Node Embeddings


