import fontTools.qu2cu.qu2cu
from torch_geometric.nn import HGTConv, HANConv
from torch_geometric.nn.inits import reset, uniform
import random
import numpy as np
import torch
from libcity.model.STRL.HGNN import HyperEncoder
from libcity.model.STRL.RoadEmb import RoadEmbedding
from libcity.model.STRL.RegionEmb import RegionEmbedding
from libcity.model.STRL.POIEmb import POIEmbedding
from libcity.model.utils import *
from torch_geometric.utils import degree
#针对单节点，多种边

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, dropout,
                 activation, node_types, metadata):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = nn.Linear(hidden_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads, group='sum')
            self.convs.append(conv)

        self.lin = nn.Linear(hidden_channels, out_channels)
        self.activation = activation

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            for node_type, x in x_dict.items():
                x_dict[node_type] = self.activation(x)

        return self.lin(x_dict['node'])

class Road2RegionAtten(nn.Module):
    def __init__(self, hidden_channels, num_regions, dropout, device, emb_dim):
        super(Road2RegionAtten, self).__init__()
        self.num_regions = num_regions
        self.emb_dim = emb_dim
        self.q_linear = nn.Linear(in_features=hidden_channels, out_features=hidden_channels, bias=False)
        self.k_linear = nn.Linear(in_features=hidden_channels, out_features=hidden_channels, bias=False)
        self.v_linear = nn.Linear(in_features=hidden_channels, out_features=hidden_channels, bias=False)
        self.road_len_length_encode_emb = nn.Embedding(200, self.emb_dim, padding_idx=0)
        self.road_degree_encode_emb = nn.Embedding(360, self.emb_dim)
        self.w1 = nn.Parameter(torch.Tensor(num_regions, self.emb_dim))
        self.w2 = nn.Parameter(torch.Tensor(num_regions, self.emb_dim))
        self.proj = nn.Linear(in_features=hidden_channels, out_features=hidden_channels, bias=False)
        self.d = torch.tensor(hidden_channels).to(device)
        self.dropout = nn.Dropout(dropout)


    def forward(self, road_emb, road2region_assign_matrix, region_fea_emb, road2region_dist_matrix, road2region_degree_matrix):
        region_emb = []
        for region in range(self.num_regions):
            region_summary = region_fea_emb[region].unsqueeze(0)  # 区域emb (1, hid, )
            road_in_region = torch.nonzero(road2region_assign_matrix[:, region] == 1).squeeze(-1)  # (N_road_in_region1)
            if len(road_in_region) == 0:
                region_emb.append(region_summary)  # (1, hid)
                continue
            degree_in_region = road2region_degree_matrix[road_in_region, region]  # (N_road_in_region1)
            degree_in_region_emb = self.road_degree_encode_emb(degree_in_region).unsqueeze(0) # (1,N_road_in_region1, hid)
            length_in_region = road2region_dist_matrix[road_in_region, region] # (N_road_in_region1)
            length_in_region_emb = self.road_len_length_encode_emb(length_in_region).unsqueeze(0)  # (1,N_road_in_region1, hid)
            part1 = torch.bmm(self.w1[region].unsqueeze(0).unsqueeze(1),
                              degree_in_region_emb.transpose(1, 2)) # (1, 1, N_road_in_region1)
            part2 = torch.bmm(self.w2[region].unsqueeze(0).unsqueeze(1),
                              length_in_region_emb.transpose(1, 2))  # (1, 1, N_road_in_region1)
            road_emb_in_region = road_emb[road_in_region].unsqueeze(0) # (1, N_road_in_region1, hid)
            query_hidden = self.q_linear(region_summary).unsqueeze(1)  # (1, 1, hid)
            key_hidden = self.k_linear(road_emb_in_region)  # (1, N_road_in_region1, hid)
            value_hidden = self.v_linear(road_emb_in_region)  # (1, N_road_in_region1, hid)
            part3 = torch.bmm(query_hidden, key_hidden.transpose(1, 2))  # (1, 1, N_road_in_region1)
            scores = (part1 + part2 + part3) / torch.sqrt(self.d) # (1, 1, N_road_in_region1)
            scores = self.dropout(torch.softmax(scores, dim=-1))  # (1, 1, N_road_in_region1)
            region_emb_i = torch.bmm(scores, value_hidden).squeeze(0)  # (1, hid)
            region_emb_i = self.proj(region_emb_i) + region_summary  # (1, hid)
            region_emb.append(region_emb_i)  # (1, hid)
        region_emb = torch.cat(region_emb, dim=0)  # (N-region, hid)
        return region_emb


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        #  seq: (N-region, hid)
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)


def corruption(x):
    return x[torch.randperm(x.size(0))]
class MLP(nn.Module):
    def __init__(self,hid_dim,num_proj_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hid_dim,num_proj_dim)
        self.fc2 = nn.Linear(num_proj_dim,hid_dim)
    def forward(self,x):
        return self.fc2(F.elu(self.fc1(x)))

class IntraEncoder(nn.Module):
    def __init__(self, config, data_feature):
        super(IntraEncoder, self).__init__()
        self.hid_dim = config['output_dim']
        self.dataset = config['dataset']
        self.device = config['device']
        self.num_proj_hidden = config['num_proj_hidden']
        self.tau = config['tau']
        self.activation = config['activation']
        self.num_layers = config['num_layers']
        self.heads = config['attn_heads']
        self.dropout = config['dropout']
        self.meta_data = data_feature['meta_data']
        self.edge_types = config['edge_types']
        self.edge_agu = config['edge_agu']
        self.fea_agu = config['fea_agu']
        self.intra_road2region = config['intra_road2region']
        self.entity_type = config['entity_type']
        self.drop_edge_rate_1 = config['drop_edge_rate_1']
        self.drop_edge_rate_2 = config['drop_edge_rate_2']
        self.drop_feature_rate_1 = config['drop_feature_rate_1']
        self.drop_feature_rate_2 = config['drop_feature_rate_2']
        if self.entity_type == 'road':
            self.emb_module = RoadEmbedding(config, data_feature)
        if self.entity_type == 'region':
            self.emb_module = RegionEmbedding(config, data_feature)
        if self.entity_type == 'poi':
            self.emb_module = POIEmbedding(config,data_feature)
        # graph encoder
        self.intra_graph_encoder = HGT(hidden_channels=self.hid_dim, out_channels=self.hid_dim, num_heads=self.heads,
                                    num_layers=self.num_layers, dropout=self.dropout,
                                    activation=get_activation(self.activation),
                                    node_types=['node'], metadata=self.meta_data)
        #self.inter_hyper_encoder = HyperEncoder(in_dim=self.hid_dim,edge_dim=self.hid_dim,node_dim=self.hid_dim)
        # projection head
        self.fc1 = nn.Linear(self.hid_dim, self.num_proj_hidden)
        self.fc2 = nn.Linear(self.num_proj_hidden, self.hid_dim)
        # self.road_fc1 = nn.Linear(self.hid_dim, self.num_proj_hidden)
        # self.road_fc2 = nn.Linear(self.num_proj_hidden, self.hid_dim)
        # self.region_fc1 = nn.Linear(self.hid_dim, self.num_proj_hidden)
        # self.region_fc2 = nn.Linear(self.num_proj_hidden, self.hid_dim)
        # self.poi_project_head1 = MLP(self.hid_dim, self.num_proj_hidden)
        # self.road_project_head1 = MLP(self.hid_dim, self.num_proj_hidden)
        # self.poi_project_head2 = MLP(self.hid_dim, self.num_proj_hidden)
        # self.road_project_head2 = MLP(self.hid_dim, self.num_proj_hidden)
        # self.region_project_head2 = MLP(self.hid_dim, self.num_proj_hidden)
        # self.set_project_head1 = MLP(self.hid_dim,self.num_proj_hidden)
        # self.set_project_head2 = MLP(self.hid_dim,self.num_proj_hidden)
        # self.mask_fill_tools = related_tool(edge=data_feature['hyperedge_index'])

        # # (num-road, num-region)
        # road2region_assign_matrix = np.load('./raw_data/{}/road2region_{}.npy'.format(self.dataset, self.dataset))
        # self.road2region_assign_matrix = torch.from_numpy(road2region_assign_matrix).to(self.device)
        #
        # road2region_degree_matrix = np.load('./raw_data/{}/degree_road2region_{}.npy'.format(self.dataset, self.dataset)).astype(np.int64)
        # self.road2region_degree_matrix = torch.from_numpy(road2region_degree_matrix).to(self.device)
        #
        # road2region_dist_matrix = np.load('./raw_data/{}/dist_road2region_{}.npy'.format(self.dataset, self.dataset)).astype(np.int64)
        # self.road2region_dist_matrix = torch.from_numpy(road2region_dist_matrix).to(self.device)
        #
        # self.road2region = Road2RegionAtten(hidden_channels=self.hid_dim,
        #                                     num_regions=self.num_regions,
        #                                     dropout=self.dropout,
        #                                     device=self.device,
        #                                     emb_dim=config['emb_dim'])
        #
        # self.region2city = AvgReadout()
        #
        # self.weight_road2region = nn.Parameter(torch.Tensor(self.hid_dim, self.hid_dim))
        # self.weight_region2city = nn.Parameter(torch.Tensor(self.hid_dim, self.hid_dim))
        # self.reset_parameters()

        self.criterion = nn.BCEWithLogitsLoss()


    def reset_parameters(self):
        reset(self.encoder)
        reset(self.road2region)
        uniform(self.hid_dim, self.weight_road2region)
        uniform(self.hid_dim, self.weight_region2city)

    def _get_data_fea_drop_weight(self, data, data_emb, item_type):
        tmp = 0
        if 'geo' in self.edge_types:
            node_deg_1_geo = degree(data[item_type, 'geo', item_type].edge_index[1], num_nodes=data_emb.shape[0])
            feature_weights_geo = feature_drop_weights_dense(data_emb, node_c=node_deg_1_geo).to(self.device)
            tmp += feature_weights_geo
        if 'sem' in self.edge_types:
            node_deg_1_sem = degree(data[item_type, 'sem', item_type].edge_index[1], num_nodes=data_emb.shape[0])
            feature_weights_sem = feature_drop_weights_dense(data_emb, node_c=node_deg_1_sem).to(self.device)
            tmp += feature_weights_sem
        if 'mob' in self.edge_types:
            node_deg_1_mob = degree(data[item_type, 'mob', item_type].edge_index[1], num_nodes=data_emb.shape[0])
            feature_weights_mob = feature_drop_weights_dense(data_emb, node_c=node_deg_1_mob).to(self.device)
            tmp += feature_weights_mob
        # feature_weights = (feature_weights_geo + feature_weights_sem + feature_weights_mob) / 3
        feature_weights = tmp / len(self.edge_types)
        return feature_weights

    def _drop_edge(self, data, item_type, p):
        if 'geo' in self.edge_types:
            drop_weights_geo = data[item_type, 'geo', item_type].edge_weight
            edge_index_geo = drop_edge_weighted(data[item_type, 'geo', item_type].edge_index,
                                                  edge_weights=drop_weights_geo, p=p, threshold=0.7)
        else:
            edge_index_geo = None
        if 'sem' in self.edge_types:
            drop_weights_sem = data[item_type, 'sem', item_type].edge_weight
            edge_index_sem = drop_edge_weighted(data[item_type, 'sem', item_type].edge_index,
                                                  edge_weights=drop_weights_sem, p=1 if item_type == 'poi_node' else p, threshold = 0.7)
        else:
            edge_index_sem = None
        if 'mob' in self.edge_types:
            drop_weights_mob = data[item_type, 'mob', item_type].edge_weight
            edge_index_mob = drop_edge_weighted(data[item_type, 'mob', item_type].edge_index,
                                                  edge_weights=drop_weights_mob, p=p, threshold=0.7)
        else:
            edge_index_mob = None
        return edge_index_geo, edge_index_sem, edge_index_mob

    def encode_poi_road_region(self, data_x, graph_edge_index, emb=True):
        if emb:
            emb_init = self.emb_module(data_x)
        else:
            emb_init = data_x
        data_input = {'node':emb_init}
        # road_embedding, region_embedding = self.encoder(data_input, data_edge_index_dict)  # (N-region, hid)
        embedding = self.intra_graph_encoder(data_input, graph_edge_index)
        embedding = F.normalize(embedding)
        return embedding

    # def forward_inter(self, data):
    #     pos_road_emb, region_emb = self.encode_road_region(data['road_node'].x, data['region_node'].x,
    #                                                        data.edge_index_dict, emb=True)  # (N-road, hid), (N-region, hid)
    #     neg_road_emb, neg_region_emb = self.encode_road_region(corruption(data['road_node'].x), data['region_node'].x,
    #                                                            data.edge_index_dict, emb=True)  # (N-road, hid), (N-region, hid)
    #     city_emb = self.region2city(region_emb)  # (hid)
    #
    #     pos_road_emb_list = []
    #     neg_road_emb_list = []
    #     for region in range(self.num_regions):
    #         road_in_region = torch.nonzero(self.road2region_assign_matrix[:, region] == 1).squeeze(-1)  # (N_road_in_region1)
    #         road_emb_in_region = pos_road_emb[road_in_region]  # (N_road_in_region1, hid)
    #         region_numbers = [num for num in range(0, self.num_regions) if num != region]
    #         another_region_id = random.choice(region_numbers)
    #         road_in_another_region = torch.nonzero(self.road2region_assign_matrix[:, another_region_id] == 1).squeeze(-1)  # (N_road_in_region2)
    #         road_emb_in_another_region = pos_road_emb[road_in_another_region]  # (N_road_in_region2, hid)
    #         pos_road_emb_list.append(road_emb_in_region)
    #         neg_road_emb_list.append(road_emb_in_another_region)
    #     return pos_road_emb_list, neg_road_emb_list, region_emb, neg_region_emb, city_emb

    def forward_intra(self, data_x, graph_edge_index):
        emb = self.encode_poi_road_region(
            data_x, graph_edge_index ,emb=False)  # (N-road, hid), (N-region, hid)
        return emb

    def forward(self, data):
        if self.intra_road2region:
            emb_init = self.encode_poi_road_region(data_x=data['node'].x,graph_edge_index=data.edge_index_dict,emb=True)  # (N-road, hid), (N-region, hid)
        else:
            emb_init = self.emb_module(data['node'].x)
        feature_drop_weights = self._get_data_fea_drop_weight(data,emb_init,'node')


        x_1 = drop_feature_weighted_2(emb_init, feature_drop_weights, self.drop_feature_rate_1,
                                           threshold=0.7)
        x_2 = drop_feature_weighted_2(emb_init, feature_drop_weights, self.drop_feature_rate_2,
                                           threshold=0.7)
        edge_index_geo_1, edge_index_sem_1, edge_index_mob_1 = \
            self._drop_edge(data, 'node', self.drop_edge_rate_1)
        edge_index_geo_2, edge_index_sem_2, edge_index_mob_2 = \
            self._drop_edge(data, 'node', self.drop_edge_rate_2)
        edge_index_1 = {}
        edge_index_2 = {}
        if 'geo' in self.edge_types:
            edge_index_1[('node', 'geo', 'node')] = edge_index_geo_1
            edge_index_2[('node', 'geo', 'node')] = edge_index_geo_2
        if 'sem' in self.edge_types:
            edge_index_1[('node', 'sem', 'node')] = edge_index_sem_1
            edge_index_2[('node', 'sem', 'node')] = edge_index_sem_2
        if 'mob' in self.edge_types:
            edge_index_1[('node', 'mob', 'node')] = edge_index_mob_1
            edge_index_2[('node', 'mob', 'node')] = edge_index_mob_2

        if self.fea_agu and self.edge_agu:
            z1 = self.forward_intra(data_x=x_1,graph_edge_index=edge_index_1)
            z2 = self.forward_intra(data_x=x_2,graph_edge_index=edge_index_2)
        elif self.fea_agu and not self.edge_agu:
            z1 = self.forward_intra(data_x=x_1,graph_edge_index=data.edge_index_dict)
            z2= self.forward_intra(data_x=x_2,graph_edge_index=data.edge_index_dict)
        elif not self.fea_agu and self.edge_agu:
            z1 = self.forward_intra(data_x=emb_init,graph_edge_index=data.edge_index_dict)
            z2 = self.forward_intra(data_x=emb_init,graph_edge_index=data.edge_index_dict)
        else:
            z1 = z2 = self.forward_intra(data_x=emb_init,graph_edge_index=edge_index_1)
        z = self.encode_poi_road_region(data_x=data['node'].x,
                                                            graph_edge_index=data.edge_index_dict,
                                                            emb=True)
        return z1,z2,z

    def projection(self, z: torch.Tensor):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)

    def intra_loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size=None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
