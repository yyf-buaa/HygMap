import numpy as np
from libcity.data.dataset import AbstractDataset
from libcity.data.dataset import HeteroRoadFeatureDataset, HeteroRegionFeatureDataset, HeteroPOIFeatureDataset
import torch
from torch_geometric.data import HeteroData
import json
import pandas as pd
from logging import getLogger


class HOMEDataset(AbstractDataset):
    def __init__(self, config):
        self._logger = getLogger()
        self.config = config
        self.dataset = self.config.get('dataset', '')  # cd
        self.device = self.config.get('device')
        self.edge_types = self.config['edge_types']
        self.road_dataset = HeteroRoadFeatureDataset(config)
        self.region_dataset = HeteroRegionFeatureDataset(config)
        self.poi_dataset = HeteroPOIFeatureDataset(config)
        region_num = len(self.region_dataset.geo_ids)
        df = pd.read_csv('/home/panda/private/jjw/raw_data/sanfransico/sanfransico.rel')
        df = df[(df['rel_type'] == 'region2road') | (df['rel_type'] == 'region2poi')].reset_index(drop=True)
        self.hyperedge_index = []
        for i in range(len(df)):
            origin_id = df.loc[i, 'origin_id']
            destin_id = df.loc[i, 'destination_id'] - region_num
            self.hyperedge_index.append([destin_id, origin_id])
        self.hyperedge_index = np.array(self.hyperedge_index).T
        self.hyperedge_index = torch.tensor(self.hyperedge_index, dtype=torch.long)
        sorted_values, indices = torch.sort(self.hyperedge_index[1, :], dim=0)
        self.hyperedge_index = self.hyperedge_index[:, indices]
        self.hyperedge_index = self.hyperedge_index.to(self.device)
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

    def drop_edge_precent(self, edge_index, edge_weight, percent):
        # edge_weight的数量
        num_edges = edge_weight.size(0)

        # 根据百分比计算要保留的边数量
        num_keep = int(num_edges * percent)

        # 根据 edge_weight 的大小进行排序（降序）
        sorted_indices = torch.argsort(edge_weight, descending=True)

        # 选择前 num_keep 个边的索引
        filtered_indices = sorted_indices[:num_keep]

        # 筛选边和对应的权重
        filtered_edge_index = edge_index[:, filtered_indices]
        filtered_edge_weight = edge_weight[filtered_indices]

        return filtered_edge_index, filtered_edge_weight

    def drop_edge_threshold(self, edge_index, edge_weight, threshold):
        # 找到满足条件的边索引
        filtered_indices = torch.where(edge_weight > threshold)[0]  # 找出大于阈值的边索引

        # 筛选边和对应的权重
        filtered_edge_index = edge_index[:, filtered_indices]
        filtered_edge_weight = edge_weight[filtered_indices]

        return filtered_edge_index, filtered_edge_weight

    def get_data(self):
        road_x = torch.from_numpy(self.road_dataset.road_features).long()
        road_edge_index_rel = torch.tensor(self.road_dataset.edge_index_rel, dtype=torch.long)
        road_edge_index_sem = torch.tensor(self.road_dataset.edge_index_sem, dtype=torch.long)
        road_edge_index_mob = torch.tensor(self.road_dataset.edge_index_mob, dtype=torch.long)
        road_edge_weight_rel = torch.from_numpy(self.road_dataset.edge_weight_rel).float()
        road_edge_weight_sem = torch.from_numpy(self.road_dataset.edge_weight_sem).float()
        road_edge_weight_mob = torch.from_numpy(self.road_dataset.edge_weight_mob).float()

        region_x = torch.from_numpy(self.region_dataset.region_features).long()
        region_edge_index_rel = torch.tensor(self.region_dataset.edge_index_rel, dtype=torch.long)
        region_edge_index_sem = torch.tensor(self.region_dataset.edge_index_sem, dtype=torch.long)
        region_edge_index_mob = torch.tensor(self.region_dataset.edge_index_mob, dtype=torch.long)
        region_edge_weight_rel = torch.from_numpy(self.region_dataset.edge_weight_rel).float()
        region_edge_weight_sem = torch.from_numpy(self.region_dataset.edge_weight_sem).float()
        region_edge_weight_mob = torch.from_numpy(self.region_dataset.edge_weight_mob).float()

        poi_x = torch.from_numpy(self.poi_dataset.poi_features).long()
        poi_edge_index_rel = torch.tensor(self.poi_dataset.edge_index_rel, dtype=torch.long)
        poi_edge_index_sem = torch.tensor(self.poi_dataset.edge_index_sem, dtype=torch.long)
        poi_edge_index_mob = torch.tensor(self.poi_dataset.edge_index_mob, dtype=torch.long)
        poi_edge_weight_rel = torch.from_numpy(self.poi_dataset.edge_weight_rel).float()
        poi_edge_weight_sem = torch.from_numpy(self.poi_dataset.edge_weight_sem).float()
        poi_edge_weight_mob = torch.from_numpy(self.poi_dataset.edge_weight_mob).float()

        # threshold = 0.5
        # road_edge_index_rel,road_edge_weight_rel = self.drop_edge(road_edge_index_rel, road_edge_weight_rel, threshold)
        # road_edge_index_sem, road_edge_weight_sem = self.drop_edge_precent(road_edge_index_sem, road_edge_weight_sem,
        #                                                                    0.05)
        # road_edge_index_mob, road_edge_weight_mob = self.drop_edge(road_edge_index_mob, road_edge_weight_mob, threshold)
        #
        # region_edge_index_rel,region_edge_weight_rel = self.drop_edge(region_edge_index_rel, region_edge_weight_rel, threshold)
        # region_edge_index_sem, region_edge_weight_sem = self.drop_edge(region_edge_index_sem, region_edge_weight_sem, threshold)
        # region_edge_index_mob, region_edge_weight_mob = self.drop_edge(region_edge_index_mob, region_edge_weight_mob, threshold)

        # poi_edge_index_rel, poi_edge_weight_rel = self.drop_edge_threshold(poi_edge_index_rel, poi_edge_weight_rel, 0.1)
        # poi_edge_index_sem, poi_edge_weight_sem = self.drop_edge_precent(poi_edge_index_sem, poi_edge_weight_sem, 0.05)
        # poi_edge_index_mob, poi_edge_weight_mob = self.drop_edge(poi_edge_index_mob, poi_edge_weight_mob, threshold)

        self._logger.info(
            'road_geo:{},road_sem:{},road_mob:{}'.format(len(road_edge_weight_rel), len(road_edge_weight_sem),
                                                         len(road_edge_weight_mob)))
        self._logger.info(
            'poi_geo:{},poi_sem:{},poi_mob:{}'.format(len(poi_edge_weight_rel), len(poi_edge_weight_sem),
                                                      len(poi_edge_weight_mob)))
        self._logger.info(
            'region_geo:{},region_sem:{},region_mob:{}'.format(len(region_edge_weight_rel), len(region_edge_weight_sem),
                                                               len(region_edge_weight_mob)))
        # region_graph = HeteroData()
        # road_graph = HeteroData()
        # poi_graph = HeteroData()
        # region_graph['region_node'].x = region_x
        # road_graph['road_node'].x = road_x
        # poi_graph['poi_node'].x = poi_x
        #
        # region_graph['region_node', 'geo', 'region_node'].edge_index = region_edge_index_rel
        # region_graph['region_node', 'geo', 'region_node'].edge_weight = region_edge_weight_rel
        # region_graph['region_node', 'sem', 'region_node'].edge_index = region_edge_index_sem
        # region_graph['region_node', 'sem', 'region_node'].edge_weight = region_edge_weight_sem
        # region_graph['region_node', 'mob', 'region_node'].edge_index = region_edge_index_mob
        # region_graph['region_node', 'mob', 'region_node'].edge_weight = region_edge_weight_mob
        #
        # road_graph['road_node', 'geo', 'road_node'].edge_index = road_edge_index_rel
        # road_graph['road_node', 'geo', 'road_node'].edge_weight = road_edge_weight_rel
        # road_graph['road_node', 'sem', 'road_node'].edge_index = road_edge_index_sem
        # road_graph['road_node', 'sem', 'road_node'].edge_weight = road_edge_weight_sem
        # road_graph['road_node', 'mob', 'road_node'].edge_index = road_edge_index_mob
        # road_graph['road_node', 'mob', 'road_node'].edge_weight = road_edge_weight_mob
        #
        # poi_graph['poi_node', 'geo', 'poi_node'].edge_index = poi_edge_index_rel
        # poi_graph['poi_node', 'geo', 'poi_node'].edge_weight = poi_edge_weight_rel
        # poi_graph['poi_node', 'sem', 'poi_node'].edge_index = poi_edge_index_sem
        # poi_graph['poi_node', 'sem', 'poi_node'].edge_weight = poi_edge_weight_sem
        # poi_graph['poi_node', 'mob', 'poi_node'].edge_index = poi_edge_index_mob
        # poi_graph['poi_node', 'mob', 'poi_node'].edge_weight = poi_edge_weight_mob
        # poi_graph = poi_graph.to(self.device)
        # region_graph = region_graph.to(self.device)
        # road_graph = road_graph.to(self.device)
        # data = {'region':region_graph,'road':road_graph,'poi':poi_graph,'hyperedge_index':self.hyperedge_index}
        # self.meta_data = {}
        # self.meta_data['region'] = region_graph.metadata()
        # self.meta_data['road'] = road_graph.metadata()
        # self.meta_data['poi'] = poi_graph.metadata()
        graph = HeteroData()
        graph['region_node'].x = region_x
        graph['road_node'].x = road_x
        graph['poi_node'].x = poi_x
        if 'geo' in self.edge_types:
            graph['region_node', 'geo', 'region_node'].edge_index = region_edge_index_rel
            graph['region_node', 'geo', 'region_node'].edge_weight = region_edge_weight_rel
            graph['road_node', 'geo', 'road_node'].edge_index = road_edge_index_rel
            graph['road_node', 'geo', 'road_node'].edge_weight = road_edge_weight_rel
            graph['poi_node', 'geo', 'poi_node'].edge_index = poi_edge_index_rel
            graph['poi_node', 'geo', 'poi_node'].edge_weight = poi_edge_weight_rel
        if 'mob' in self.edge_types:
            graph['region_node', 'mob', 'region_node'].edge_index = region_edge_index_mob
            graph['region_node', 'mob', 'region_node'].edge_weight = region_edge_weight_mob
            graph['road_node', 'mob', 'road_node'].edge_index = road_edge_index_mob
            graph['road_node', 'mob', 'road_node'].edge_weight = road_edge_weight_mob
            graph['poi_node', 'mob', 'poi_node'].edge_index = poi_edge_index_mob
            graph['poi_node', 'mob', 'poi_node'].edge_weight = poi_edge_weight_mob
        if 'sem' in self.edge_types:
            graph['region_node', 'sem', 'region_node'].edge_index = region_edge_index_sem
            graph['region_node', 'sem', 'region_node'].edge_weight = region_edge_weight_sem
            graph['road_node', 'sem', 'road_node'].edge_index = road_edge_index_sem
            graph['road_node', 'sem', 'road_node'].edge_weight = road_edge_weight_sem
            graph['poi_node', 'sem', 'poi_node'].edge_index = poi_edge_index_sem
            graph['poi_node', 'sem', 'poi_node'].edge_weight = poi_edge_weight_sem
        graph = graph.to(self.device)
        self.meta_data = graph.metadata()
        data = {'intra_graph': graph, 'hyperedge_index': self.hyperedge_index}
        return data, None, None

    def get_data_feature(self):
        res = {}
        res['meta_data'] = self.meta_data
        res.update(self.road_dataset.get_data_feature())
        res.update(self.region_dataset.get_data_feature())
        res.update(self.poi_dataset.get_data_feature())
        res['hyperedge_index'] = self.hyperedge_index
        return res
