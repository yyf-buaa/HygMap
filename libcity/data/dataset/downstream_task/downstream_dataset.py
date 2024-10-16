from abc import ABC, abstractmethod
import datetime
import json
import logging
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DownstreamDataset(ABC, Dataset):
    def __init__(self, args, file_path, embedding_path):
        self.file_path = file_path
        self.max_len = args.max_len
        self.args = args
        self.cache_path_feature = os.path.join(args.data_cache_path,
                                               Path(file_path).stem + f'_{args.model_name}_v2.pkl')
        self.cache_path_wkt = os.path.join(args.data_cache_path, Path(file_path).stem + f'_{args.model_name}_wkt.json')
        self.embedding_path = embedding_path
        self.embedding = np.load(embedding_path)
        self._logger = logging.getLogger()
        self._logger.info('finish embedding loading:shape = '+str(self.embedding.shape))
        self._load_geo_latlon()
        self.load(file_path)

    @abstractmethod
    def process_traj(self):
        pass

    def _load_geo_latlon(self):
        self.geo_file = pd.read_csv('raw_data/{}/roadmap_{}/roadmap_{}.geo'.format(self.args.dataset, self.args.dataset,self.args.dataset))
        assert self.geo_file['type'][0] == 'LineString'
        self.geo2latlon = {}
        for i in range(self.geo_file.shape[0]):
            geo_id = int(self.geo_file.iloc[i]['geo_id'])
            coordinates = eval(self.geo_file.iloc[i]['coordinates'])
            self.geo2latlon[geo_id] = coordinates
        self._logger.info("Loaded Geo2LatLon, num_nodes=" + str(len(self.geo2latlon)))

    def load(self, file_path):
        if self.args.use_cache and os.path.exists(self.cache_path_feature) and os.path.exists(self.cache_path_wkt):
            self.trajs, self.times, self.ids, self.lengths = pickle.load(open(self.cache_path_feature, 'rb'))
            self._logger.info('read cached')
        else:
            self.trajs = []
            self.times = []
            self.uids = []
            self.ids = []
            self.lengths = []
            traj_wkt = {}
            data = pd.read_csv(file_path, sep=';', dtype={'id': int, 'hop': int, 'traj_id': int})

            paths = data['path'].values
            times = data['tlist'].values
            #usrlist = data['usr_id'].values
            idlist = data['id'].values
            lenlist = data['length'].values
            # stream = open(file_path, 'r')
            for i in tqdm(range(len(paths)), desc=f'Loading Data {file_path}'):
                roads = paths[i]
                tlist = times[i]
                #usrid = int(usrlist[i])
                id_ = int(idlist[i])
                length = float(lenlist[i])
                roads = roads.strip('[').strip(']').split(', ')[:self.max_len]
                roads = [int(road) for road in roads]
                tlist = tlist.strip('[').strip(']').split(', ')[:self.max_len]
                tlist = [int(time) for time in tlist]

                # cal wkt str
                wkt_str = 'LINESTRING('
                for j in range(len(roads)):
                    rid = roads[j]
                    coordinates = self.geo2latlon.get(int(rid), [])  # [(lat1, lon1), (lat2, lon2), ...]
                    if not coordinates:
                        # Occur when vocab is not full coverage
                        print(rid)
                    for coor in coordinates:
                        wkt_str += (str(coor[0]) + ' ' + str(coor[1]) + ',')
                if wkt_str[-1] == ',':
                    wkt_str = wkt_str[:-1]
                wkt_str += ')'
                traj_wkt[id_] = wkt_str
                self.trajs.append(roads)
                self.times.append(tlist)
                #self.uids.append(usrid)
                self.ids.append(id_)
                self.lengths.append(length)
            pickle.dump([self.trajs, self.times,self.ids, self.lengths],
                        open(self.cache_path_feature, 'wb'))
            json.dump(traj_wkt, open(self.cache_path_wkt, 'w'))
        print("load trajectory completed")

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, item):
        traj = self.trajs[item]
        tlist = self.times[item]
        trajid = self.ids[item]
        length = self.lengths[item]
        last_traj, last_time, traj, tlist = self.process_traj(traj, tlist, self.max_len)
        dura_time = float(int(last_time) - int(tlist[0])) / 60

        traj_input = traj
        tlist = [int(tim) for tim in tlist]
        new_tim_list = [datetime.datetime.utcfromtimestamp(tim) for tim in tlist]

        traj_input = traj_input
        input_mask = [1] * len(traj_input)
        hop = [len(traj_input)]
        traj_emb = self.embedding[traj_input]
        traj_emb = np.pad(traj_emb, ((0, self.max_len-hop[0]), (0, 0)), mode='constant')
        padding = [0 for _ in range(self.max_len - len(traj_input))]

        tlist.extend(padding)
        traj_input.extend(padding)
        input_mask.extend(padding)

        output = {"road_emb":traj_emb,"traj_input": traj_input, "input_mask": input_mask, 'hop': hop,
                  "eta_target": dura_time, "pos_target": last_traj,
                  "traj_id": trajid, "time_list": tlist, 'length': length}

        return {key: torch.tensor(value) for key, value in output.items()}