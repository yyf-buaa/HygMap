import os
import pandas as pd
import numpy as np
from logging import getLogger
from libcity.data.dataset import AbstractDataset


class NetWorkDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')  # cd
        self.type = self.config.get('type', 'road')
        self.linktype = self.config.get('linktype', 'rel')
        if self.type == 'road':
            self.data_path = './raw_data/{}/roadmap_{}/roadmap_'.format(self.dataset, self.dataset)
        elif self.type == 'region':
            self.data_path = './raw_data/{}/regionmap_{}/regionmap_'.format(self.dataset, self.dataset)
        else:
            raise ValueError('Error type {}'.format(self.type))
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.bidir_adj_mx = self.config.get('bidir_adj_mx', False)
        self._logger = getLogger()
        assert os.path.exists(self.data_path + self.geo_file + '.geo')
        assert os.path.exists(self.data_path + self.rel_file + '.rel')
        self.geo_file = self._load_geo()
        self.rel_file = self._load_rel()

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        for index, geo_id in enumerate(self.geo_ids):
            self.geo_to_ind[geo_id] = index
            self.ind_to_geo[index] = geo_id
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_ids)))
        return geofile

    def _load_rel(self):
        """
        加载.rel文件，格式[rel_id, type, origin_id, destination_id, properties(若干列)],
        生成N*N的邻接矩阵，

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')[['origin_id', 'destination_id']]
        # 把数据转换成矩阵的形式
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        for row in relfile.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
            if self.bidir_adj_mx:
                self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = 1
        self._logger.info("Loaded file " + self.rel_file + '.rel, shape=' + str(self.adj_mx.shape) +
                          ', edges=' + str(self.adj_mx.sum()))
        return relfile

    def get_data(self):
        """
        返回数据的DataLoader，此类只负责返回路网结构adj_mx，而adj_mx在data_feature中，这里什么都不返回
        """
        return None, None, None

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"adj_mx": self.adj_mx, "num_nodes": self.num_nodes,
                "geo_file": self.geo_file, "rel_file": self.rel_file,
                "geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo}
