import numpy as np
from libcity.data.dataset import AbstractDataset
from libcity.data.dataset import HeteroRoadFeatureDataset, HeteroRegionFeatureDataset, HeteroPOIFeatureDataset
import torch
from torch_geometric.data import HeteroData
import json
import pandas as pd
from logging import getLogger


class InterDataset(AbstractDataset):
    def __init__(self, config):
        self._logger = getLogger()
        self.config = config
        self.dataset = self.config.get('dataset', '')  # cd
        self.device = self.config.get('device')
        self.road_embedding_path = self.config.get('road_embedding_path')
        self.region_embedding_path = self.config.get('region_embedding_path')
        self.poi_embedding_path = self.config.get('poi_embedding_path')
        self.road_embedding = np.load(self.road_embedding_path)
        self.region_embedding = np.load(self.region_embedding_path)
        self.poi_embedding = np.load(self.poi_embedding_path)
        self.road_embedding = torch.from_numpy(self.road_embedding).to(self.device)
        self.region_embedding = torch.from_numpy(self.region_embedding).to(self.device)
        self.poi_embedding = torch.from_numpy(self.poi_embedding).to(self.device)
        region_num = self.region_embedding.shape[0]
        df = pd.read_csv("/home/panda/private/jjw/raw_data/sanfransico/sanfransico.rel")
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
    def get_data(self):
        data = {'hyperedge_index': self.hyperedge_index,'road_embedding':self.road_embedding,'region_embedding':self.region_embedding, 'poi_embedding':self.poi_embedding}
        return data, None, None

    def get_data_feature(self):
        res = {'hyperedge_index': self.hyperedge_index,'road_embedding':self.road_embedding,'region_embedding':self.region_embedding, 'poi_embedding':self.poi_embedding}
        return res
