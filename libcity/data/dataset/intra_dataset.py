import numpy as np
from libcity.data.dataset import AbstractDataset
from libcity.data.dataset import HeteroRoadFeatureDataset, HeteroRegionFeatureDataset, HeteroPOIFeatureDataset
import torch
from torch_geometric.data import HeteroData
import json
import pandas as pd
from logging import getLogger


class IntraDataset(AbstractDataset):
    def __init__(self, config):
        self._logger = getLogger()
        self.config = config
        self.dataset = self.config.get('dataset', '')  # cd
        self.device = self.config.get('device')
        self.edge_types = self.config['edge_types']
        self.entity_type = self.config['entity_type']
        if self.entity_type == 'road':
            self.intra_dataset = HeteroRoadFeatureDataset(config)
        if self.entity_type == 'region':
            self.intra_dataset = HeteroRegionFeatureDataset(config)
        if self.entity_type == 'poi':
            self.intra_dataset = HeteroPOIFeatureDataset(config)
        # region_num = len(self.region_dataset.geo_ids)
        # df = pd.read_csv('/home/panda/private/jjw/raw_data/sanfransico/sanfransico.rel')
        # df = df[(df['rel_type'] == 'region2road') | (df['rel_type'] == 'region2poi')].reset_index(drop=True)
        # self.hyperedge_index = []
        # for i in range(len(df)):
        #     origin_id = df.loc[i, 'origin_id']
        #     destin_id = df.loc[i, 'destination_id'] - region_num
        #     self.hyperedge_index.append([destin_id, origin_id])
        # self.hyperedge_index = np.array(self.hyperedge_index).T
        # self.hyperedge_index = torch.tensor(self.hyperedge_index, dtype=torch.long)
        # sorted_values, indices = torch.sort(self.hyperedge_index[1, :], dim=0)
        # self.hyperedge_index = self.hyperedge_index[:, indices]
        # self.hyperedge_index = self.hyperedge_index.to(self.device)
        # road2region = json.load(open('./raw_data/{}/road2region_{}.json'.format(self.dataset, self.dataset), 'r'))
        # region2road = json.load(open('./raw_data/{}/region2road_{}.json'.format(self.dataset, self.dataset), 'r'))
        #
        # self.road2region_edge_index = []
        # for k, v in road2region.items():
        #     if isinstance(v, list):
        #         for vi in v:
        #             self.road2region_edge_index.append([int(k), vi])
        #     else:
        #         self.road2region_edge_index.append([int(k), v])
        # self.road2region_edge_index = np.array(self.road2region_edge_index).T
        # self.region2road_edge_index = []
        # for k, v in region2road.items():
        #     if isinstance(v, list):
        #         for vi in v:
        #             self.region2road_edge_index.append([int(k), vi])
        #     else:
        #         self.region2road_edge_index.append([int(k), v])
        # self.region2road_edge_index = np.array(self.region2road_edge_index).T

    import torch

    import torch

    # def drop_edge_precent(self, edge_index, edge_weight, percent):
    #     # edge_weight的数量
    #     num_edges = edge_weight.size(0)
    #
    #     # 根据百分比计算要保留的边数量
    #     num_keep = int(num_edges * percent)
    #
    #     # 根据 edge_weight 的大小进行排序（降序）
    #     sorted_indices = torch.argsort(edge_weight, descending=True)
    #
    #     # 选择前 num_keep 个边的索引
    #     filtered_indices = sorted_indices[:num_keep]
    #
    #     # 筛选边和对应的权重
    #     filtered_edge_index = edge_index[:, filtered_indices]
    #     filtered_edge_weight = edge_weight[filtered_indices]
    #
    #     return filtered_edge_index, filtered_edge_weight
    #
    # def drop_edge_threshold(self, edge_index, edge_weight, threshold):
    #     # 找到满足条件的边索引
    #     filtered_indices = torch.where(edge_weight > threshold)[0]  # 找出大于阈值的边索引
    #
    #     # 筛选边和对应的权重
    #     filtered_edge_index = edge_index[:, filtered_indices]
    #     filtered_edge_weight = edge_weight[filtered_indices]
    #
    #     return filtered_edge_index, filtered_edge_weight

    def get_data(self):
        return self.intra_dataset.get_data()

    def get_data_feature(self):
        return self.intra_dataset.get_data_feature()
