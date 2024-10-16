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

        return self.lin(x_dict['poi_node']), self.lin(x_dict['road_node']), self.lin(x_dict['region_node']),

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

class InterEncoder(nn.Module):
    def __init__(self, config, data_feature):
        super(InterEncoder, self).__init__()
        self.hid_dim = config['output_dim']
        self.dataset = config['dataset']
        self.device = config['device']
        self.num_proj_hidden = config['num_proj_hidden']
        self.tau = config['tau']
        self.activation = config['activation']
        self.num_layers = config['num_layers']
        self.heads = config['attn_heads']
        self.dropout = config['dropout']
        self.inter_hyper_encoder = HyperEncoder(in_dim=self.hid_dim,edge_dim=self.hid_dim,node_dim=self.hid_dim)
        self.decoder = MLP(self.hid_dim,self.hid_dim)
        # self.road_decoder = MLP(self.hid_dim,self.hid_dim)
        # self.poi_decoder = MLP(self.hid_dim, self.hid_dim)
        self.criterion = nn.BCEWithLogitsLoss()
        self.mask_poi_token = nn.Parameter(torch.rand(1,self.hid_dim)).to(self.device)
        self.mask_road_token = nn.Parameter(torch.rand(1, self.hid_dim)).to(self.device)
        self.mask_region_token = nn.Parameter(torch.rand(1, self.hid_dim)).to(self.device)
    def encoding_mask_noise(self, x, enc_mask_token, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        # if self._replace_rate > 0:
        #     num_noise_nodes = int(self._replace_rate * num_mask_nodes)
        #     perm_mask = torch.randperm(num_mask_nodes, device=x.device)
        #     token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
        #     noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
        #     noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
        #
        #     out_x = x.clone()
        #     out_x[token_nodes] = 0.0
        #     out_x[noise_nodes] = x[noise_to_be_chosen]
        # else:
        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[mask_nodes] += enc_mask_token

        return out_x, mask_nodes, keep_nodes

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.road2region)
        uniform(self.hid_dim, self.weight_road2region)
        uniform(self.hid_dim, self.weight_region2city)
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

    def forward_intra(self, data_poi_x, data_road_x, data_region_x, graph_edge_index ,hyperedge_index):
        poi_emb, road_emb, region_emb = self.encode_poi_road_region(
            data_poi_x, data_road_x, data_region_x, graph_edge_index,hyperedge_index, emb=False)  # (N-road, hid), (N-region, hid)
        return poi_emb, road_emb, region_emb

    def encode_hypergraph(self,road_embedding, poi_embedding, region_embedding, hyperedge_index):
        # poi_embedding = self.poi_ffn(poi_embedding)
        # road_embedding = self.road_ffn(road_embedding)
        road_num = road_embedding.shape[0]
        # region_embedding = self.region_ffn(region_embedding)
        hyperNode_embedding = torch.cat((road_embedding, poi_embedding), dim=0)
        hyperedge_embedding = region_embedding
        hyperNode_embedding, hyperedge_embedding = self.inter_hyper_encoder(x=hyperNode_embedding,
                                                                            e=hyperedge_embedding,
                                                                            hyperedge_index=hyperedge_index)
        road_embedding = hyperNode_embedding[:road_num, :]
        poi_embedding = hyperNode_embedding[road_num:, :]
        region_embedding = hyperedge_embedding
        region_embedding = torch.nn.functional.normalize(region_embedding, p=2.0, dim=1, eps=1e-12, out=None)
        road_embedding = torch.nn.functional.normalize(road_embedding, p=2.0, dim=1, eps=1e-12, out=None)
        poi_embedding = torch.nn.functional.normalize(poi_embedding, p=2.0, dim=1, eps=1e-12, out=None)
        return poi_embedding, road_embedding, region_embedding

    def forward(self, data):
        poi_embedding_mask,mask_poi,keep_poi = self.encoding_mask_noise(data['poi_embedding'],self.mask_poi_token,0.3)
        road_embedding_mask,mask_road,keep_road = self.encoding_mask_noise(data['road_embedding'],self.mask_road_token,0.3)
        region_embedding_mask, mask_region, keep_region = self.encoding_mask_noise(data['region_embedding'],
                                                                                 self.mask_region_token, 0.3)
        poi_z1,road_z1,region_z1 = self.encode_hypergraph(road_embedding_mask,poi_embedding_mask,data['region_embedding'],data['hyperedge_index'])
        poi_z2, road_z2, region_z2 = self.encode_hypergraph(data['road_embedding'], data['poi_embedding'],
                                                                 region_embedding_mask, data['hyperedge_index'])
        road_result = self.decoder(road_z1[mask_road])
        region_result = self.decoder(region_z2[mask_region])
        poi_result = self.decoder(poi_z1[mask_poi])
        return road_result,data['road_embedding'][mask_road], region_result,data['region_embedding'][mask_region], poi_result,data['poi_embedding'][mask_poi]

    def return_loss(self,road_x,road_y,region_x,region_y,poi_x,poi_y):
        return self.sce_loss(road_x,road_y,2), self.sce_loss(region_x,region_y,2), self.sce_loss(poi_x,poi_y,2)

    def sce_loss(self, x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        # loss =  - (x * y).sum(dim=-1)
        # loss = (x_h - y_h).norm(dim=1).pow(alpha)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss


    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        z1 = z1.to('cuda:2')
        z2 = z2.to('cuda:2')
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
        result = torch.cat(losses)
        z1 = z1.to('cuda:0')
        z2 = z2.to('cuda:0')
        result = result.to('cuda:0')
        return result

    def road_loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size=None):
        h1 = self.road_projection(z1)
        h2 = self.road_projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def region_loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size=None):
        h1 = self.region_projection(z1)
        h2 = self.region_projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def poi_loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size=None):
        h1 = self.poi_projection(z1)
        h2 = self.poi_projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    # def discriminate_road2region(self, road_emb_list, region_emb, sigmoid=True):
    #     values = []
    #     for region_id, road_emb_in_region in enumerate(road_emb_list):  # region_id 区域索引, road_emb_in_region 区域内的road-emb的数组 (N_road_in_region1, hid)
    #         if road_emb_in_region.size()[0] > 0:
    #             region_summary = region_emb[region_id]   # 区域emb (hid, )
    #             value = torch.matmul(road_emb_in_region, torch.matmul(self.weight_road2region, region_summary))  # (N_road_in_region1,)
    #             values.append(value)
    #     values = torch.cat(values, dim=0)  # (All_roads_selected)
    #     return torch.sigmoid(values) if sigmoid else values
    #
    # def road_region_loss(self, pos_road_emb_list, neg_road_emb_list, region_emb):
    #
    #     pos_scores = self.discriminate_road2region(pos_road_emb_list, region_emb, sigmoid=False)
    #     pos_loss = self.criterion(pos_scores, torch.ones_like(pos_scores))
    #
    #     neg_scores = self.discriminate_road2region(neg_road_emb_list, region_emb, sigmoid=False)
    #     neg_loss = self.criterion(neg_scores, torch.zeros_like(neg_scores))
    #
    #     loss_road2region = pos_loss + neg_loss
    #     return loss_road2region
    #
    # def discriminate_region2city(self, region_emb, city_emb, sigmoid=True):
    #     # region_emb (N-region, hid), city_emb(hid)
    #     value = torch.matmul(region_emb, torch.matmul(self.weight_region2city, city_emb))
    #     return torch.sigmoid(value) if sigmoid else value  # (N-region,)
    #
    # def region_city_loss(self, region_emb, neg_region_emb, city_emb):
    #     pos_scores = self.discriminate_region2city(region_emb, city_emb, sigmoid=False)
    #     pos_loss = self.criterion(pos_scores, torch.ones_like(pos_scores))
    #
    #     neg_scores = self.discriminate_region2city(neg_region_emb, city_emb, sigmoid=False)
    #     neg_loss = self.criterion(neg_scores, torch.zeros_like(neg_scores))
    #
    #     loss_region2city = pos_loss + neg_loss
    #     return loss_region2city

    #在超图上算inter_level的loss,masked-fill
    def mask_fill_loss(self, poi_embedding, road_embedding, head):
        poi_embedding = self.poi_project_head1(poi_embedding)
        road_embedding = self.road_project_head1(road_embedding)
        return self.mask_fill_tools.return_loss(poi_embedding,road_embedding,head)
    def form_region_loss(self, poi_embedding, road_embedding , region_embedding, head):
        poi_embedding = self.poi_project_head2(poi_embedding)
        road_embedding = self.road_project_head2(road_embedding)
        region_embedding = self.region_project_head2(region_embedding)
        return self.mask_fill_tools.return_loss_region(poi_embedding,road_embedding,region_embedding,head)
class related_tool():

    def __init__(self, edge):

        ## Create pre-requisite materials for hyperedge filling
        self.device = 'cuda:1'
        edge_index = edge.to('cpu').numpy().copy()
        self.edges = edge.clone().to(self.device)
        self.edge_num = self.edges[1][-1]+1
        # self.edges = self.edges.to(device)
        # self.device = device
        self.pairwise_srcs = []
        self.pairwise_targets = []

        e_idx = 0
        prev_indptr = 0

        self.edge_dict = dict()
        self.totalE = []

        for i in range(edge_index.shape[1]):
            cur_edge_index = edge_index[1][i]
            if e_idx != cur_edge_index:
                if i - prev_indptr > 1:  # Size is greater than 2
                    self.edge_dict[e_idx] = (list(edge_index[0][prev_indptr: i]))
                    self.totalE.append(list(edge_index[0][prev_indptr: i]))

                e_idx = cur_edge_index
                prev_indptr = i

        if i - prev_indptr > 1:  # Size is greater than 2
            self.edge_dict[e_idx] = (list(edge_index[0][prev_indptr: i + 1]))
            self.totalE.append(list(edge_index[0][prev_indptr: i + 1]))

        eidx = 0
        n_total_negs = 0
        pushVs = []
        pushIDX = []
        pullsrc = []

        for i, e in enumerate(self.totalE):

            Vs = e

            for k in range(len(Vs)):
                pushVs.extend(Vs[:k] + Vs[(k + 1):])
                pushIDX.extend([eidx] * (len(Vs) - 1))
                pullsrc.append(Vs[k])
                eidx += 1

        self.target = torch.tensor(pushIDX).to(self.device)
        self.pushVs = pushVs
        self.pullsrc = pullsrc

        self.batch = True
        self.total_target = []
        self.total_pushVs = []
        self.total_pullsrc = []


        batch_size = 10

        n_batch = (len(self.totalE) // batch_size) + 1

        for idx in range(n_batch):

            t_start, t_end = int(idx * batch_size), int((idx + 1) * batch_size)
            pushVs = []
            pushIDX = []
            pullsrc = []
            eidx = 0

            for i, e in enumerate(self.totalE[t_start:t_end]):

                Vs = e

                for k in range(len(Vs)):
                    pushVs.extend(Vs[:k] + Vs[(k + 1):])
                    pushIDX.extend([eidx] * (len(Vs) - 1))
                    pullsrc.append(Vs[k])
                    eidx += 1

            target = torch.tensor(pushIDX).to(self.device)
            self.total_target.append(target)
            self.total_pushVs.append(pushVs)
            self.total_pullsrc.append(pullsrc)

        # else:
        #     self.batch = False
        #     pushVs = []
        #     pushIDX = []
        #     pullsrc = []
        #
        #     for i, e in enumerate(self.totalE):
        #
        #         Vs = e
        #
        #         for k in range(len(Vs)):
        #             pushVs.extend(Vs[:k] + Vs[(k + 1):])
        #             pushIDX.extend([eidx] * (len(Vs) - 1))
        #             pullsrc.append(Vs[k])
        #             eidx += 1
        #
        #     self.target = torch.tensor(pushIDX).to(self.device)
        #     self.pushVs = pushVs
        #     self.pullsrc = pullsrc
    def return_loss_region(self, poi_embedding, road_embedding, region_embedding,head):
        poi_embedding = poi_embedding.to('cuda:1')
        road_embedding = road_embedding.to('cuda:1')
        region_embedding = region_embedding.to('cuda:1')
        loss1 = 0
        loss2 = 0
        Z = torch.cat((road_embedding, poi_embedding), dim=0)
        region_embedding = torch.nn.functional.normalize(region_embedding, p=2.0, dim=1, eps=1e-12, out=None)
        aggZ = head(scatter(src=Z[self.edges[0],:], index=self.edges[1], reduce='mean' , dim = 0))
        aggZ = torch.nn.functional.normalize(aggZ, p=2.0, dim=1, eps=1e-12, out=None)
        denom = torch.mm(aggZ,region_embedding.transpose(1,0))
        loss1 += -torch.sum(denom[range(self.edge_num),range(self.edge_num)])
        loss2 += torch.sum(torch.logsumexp(denom, dim=1))
        return (loss1 + loss2).to('cuda:0')

    def return_loss(self, poi_embedding , road_embedding, head):
        poi_embedding = poi_embedding.to('cuda:1')
        road_embedding = road_embedding.to('cuda:1')
        loss1 = 0
        loss2 = 0
        Z = torch.cat((road_embedding, poi_embedding), dim=0)

        normZ = torch.nn.functional.normalize(Z, p=2.0, dim=1, eps=1e-12, out=None)

        for i in range(len(self.total_target)):
            pushVs = self.total_pushVs[i]
            target = self.total_target[i]
            pullsrc = self.total_pullsrc[i]
            aggZ = head(scatter(src=Z[pushVs, :], index=target, reduce='mean', dim=0))
            aggZ = torch.nn.functional.normalize(aggZ, p=2.0, dim=1, eps=1e-12, out=None)
            denom = torch.mm(aggZ, normZ.transpose(1, 0))
            loss1 += -torch.sum(denom[range(len(pullsrc)), pullsrc])
            loss2 += torch.sum(torch.logsumexp(denom, dim=1))
        return (loss1 + loss2).to('cuda:0')