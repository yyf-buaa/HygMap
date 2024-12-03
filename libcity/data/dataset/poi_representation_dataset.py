import math
import os
import ast
import numpy as np
import pandas as pd
from logging import getLogger
from random import *
from datetime import datetime
import pickle
import pdb


class POIRepresentationDataset(object):

    def __init__(self, config):
        """
        @param raw_df: raw DataFrame containing all mobile signaling records.
            Should have at least three columns: user_id, latlng and datetime.
        @param coor_df: DataFrame containing coordinate information.
            With an index corresponding to latlng, and two columns: lat and lng.
        """
        self.config = config
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        os.makedirs(self.cache_file_folder, exist_ok=True)
        self._logger = getLogger()
        self._logger.info('Starting load data ...')
        self.cache=self.config.get('cache',True)
        self.dataset = self.config.get('dataset')
        self.test_scale = self.config.get('test_scale', 0.4)
        self.min_len = self.config.get('poi_min_len', 5)  # 轨迹最短长度
        self.min_frequency = self.config.get('poi_min_frequency', 10)  # POI 最小出现次数
        self.min_poi_cnt = self.config.get('poi_min_poi_cnt', 50)  # 用户最少拥有 POI 数
        self.pre_len = self.config.get('pre_len', 3)  # 预测后 pre_len 个 POI
        self.min_sessions = self.config.get('min_sessions', 3)# 每个user最少的session数
        self.time_threshold = self.config.get('time_threshold', 24)# 超过24小时就切分,暂时
        self.cut_method = self.config.get('cut_method','time_interval') # time_interval, same_day, fix_len
        self.w2v_window_size = self.config.get('w2v_window_size', 1)
        self.max_seq_len=self.config.get('poi_max_seq_len',32)
        self.data_path = './raw_data/' + self.dataset + '/'
        self.offset = 0
        self.cache_file_name = os.path.join(self.cache_file_folder,
                                            f'cache_{self.dataset}_{self.cut_method}_{self.max_seq_len}_{self.min_len}_{self.min_frequency}_{self.min_poi_cnt}_{self.pre_len}_{self.min_sessions}_{self.time_threshold}.pickle')

        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))
        self.usr_file = self.config.get('usr_file', self.dataset)
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        if os.path.exists(os.path.join(self.data_path, self.geo_file + '.geo')):
            self._load_geo()
        else:
            raise ValueError('Not found .geo file!')
        if os.path.exists(os.path.join(self.data_path, self.dyna_file + '.citraj')):
            self._load_dyna()
        else:
            raise ValueError('Not found .citraj file!')

        if self.cache and os.path.exists(self.cache_file_name):
            self._logger.info(f'load data from cache file: {self.cache_file_name}')
            with open(self.cache_file_name,'rb') as f:
                self.train_set=pickle.load(f)
                self.test_set=pickle.load(f)
                self.w2v_data=pickle.load(f)
                self.loc_index_map=pickle.load(f)
                self.user_index_map=pickle.load(f)
                loc_index_map=self.loc_index_map
                user_index_map=self.user_index_map
                self.max_seq_len=pickle.load(f)
                self.df = self.df[self.df['user_index'].isin(self.user_index_map)]
                self.df = self.df[self.df['loc_index'].isin(self.loc_index_map)]
                self.coor_df = self.coor_df[self.coor_df['geo_id'].isin(loc_index_map)]
                self.df['user_index'] = self.df['user_index'].map(user_index_map)
                self.coor_df['geo_id'] = self.coor_df['geo_id'].map(loc_index_map)
                self.df['loc_index'] = self.df['loc_index'].map(loc_index_map)
                self.num_user = len(user_index_map)
                self.num_loc = self.coor_df.shape[0]
                self.coor_mat = self.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index').to_numpy()
                self.id2coor_df = self.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index'). \
                    set_index('loc_index').sort_index()
                
        else:
            self.res=self.cutter_filter()
            self._init_data_feature()

        self._logger.info('User num: {}'.format(self.num_user))
        self._logger.info('Location num: {}'.format(self.num_loc))
        self._logger.info('Total checkins: {}'.format(self.df.shape[0]))
        self._logger.info('Train set: {}'.format(len(self.train_set)))
        self._logger.info('Test set: {}'.format(len(self.test_set)))
        self.con = self.config.get('con', 7e8)
        self.theta = self.num_user * self.num_loc / self.con
        
    def _load_geo(self):
        geo_df = pd.read_csv(os.path.join(self.data_path, self.geo_file + '.geo'),low_memory=False)
        geo_df = geo_df[geo_df['type'] == 'Point']
        self.offset = geo_df['geo_id'].min()
        poi_list = geo_df['coordinates'].tolist()
        lng_list = []
        lat_list = []
        for s in poi_list:
            lng, lat = ast.literal_eval(s)
            lng_list.append(lng)
            lat_list.append(lat)
        lng_col = pd.Series(lng_list, name='lng')
        lat_col = pd.Series(lat_list, name='lat')
        idx_col = pd.Series(list(range(len(geo_df))), name='geo_id')
        type_name = self.config.get('poi_type_name', None)
        if type_name is not None:
            category_list=list(geo_df[type_name].drop_duplicates())
            c2i={name:i for i,name in enumerate(category_list)}
            cid_list=[]
            for name in list(geo_df[type_name]):
                cid_list.append(c2i[name])
            cid_list=pd.Series(cid_list,name='category')
            self.coor_df = pd.concat([idx_col, lat_col, lng_col, cid_list], axis=1)
        else:
            self.coor_df = pd.concat([idx_col, lat_col, lng_col], axis=1)

    def _load_dyna(self):
        dyna_df = pd.read_csv(os.path.join(self.data_path, self.dyna_file + '.citraj'))
        dyna_df = dyna_df[dyna_df['type'] == 'trajectory']
        dyna_df = dyna_df.merge(self.coor_df, left_on='location', right_on='geo_id', how='left')
        dyna_df.rename(columns={'time': 'datetime'}, inplace=True)
        dyna_df.rename(columns={'location': 'loc_index'}, inplace=True)
        dyna_df.rename(columns={'entity_id': 'user_index'}, inplace=True)
        self.df = dyna_df[['user_index', 'loc_index', 'datetime', 'lat', 'lng']]
        user_counts = self.df['user_index'].value_counts()
        self.df = self.df[self.df['user_index'].isin(user_counts.index[user_counts >= self.min_poi_cnt])]
        loc_counts = self.df['loc_index'].value_counts()
        self.coor_df = self.coor_df[self.coor_df['geo_id'].isin(loc_counts.index[loc_counts >= self.min_frequency])]
        self.df = self.df[self.df['loc_index'].isin(loc_counts.index[loc_counts >= self.min_frequency])]

    def _split_days(self):
        data = pd.DataFrame(self.df, copy=True)

        data['datetime'] = pd.to_datetime(data["datetime"])
        data['nyr'] = data['datetime'].apply(lambda x: datetime.fromtimestamp(x.timestamp()).strftime("%Y-%m-%d"))

        days = sorted(data['nyr'].drop_duplicates().to_list())
        num_days = None #self.config.get('num_days', 20)
        if num_days is not None:
            days = days[:num_days]
        if len(days) <= 1:
            raise ValueError('Dataset contains only one day!')
        test_count = max(1, min(math.ceil(len(days) * self.test_scale), len(days)))
        self.split_days = [days[:-test_count], days[-test_count:]]
        self._logger.info('Days for train: {}'.format(self.split_days[0]))
        self._logger.info('Days for test: {}'.format(self.split_days[1]))

    def _load_usr(self):
        pass

    def gen_index_map(self, df, column, offset=0):
        index_map = {origin: index + offset
                     for index, origin in enumerate(df[column].drop_duplicates())}
        return index_map

    def _init_data_feature(self):
        # 变换user的id
        # 划分train、test数据集
        # 生成seq
        # self.max_seq_len = 0
        #one_set = [user_index, 'loc_index', 'weekday'list(),'timestamp'.tolist(), loc_length]
        self.train_set = []
        self.test_set = []
        self.w2v_data = []
        # 这里之后将不会变换，所以可以进行映射了
        u_list=self.res.keys()
        self.df=self.df[self.df['user_index'].isin(u_list)]
        loc_keys = self.df['loc_index'].value_counts().keys()
        self.coor_df=self.coor_df[self.coor_df['geo_id'].isin(loc_keys)]
        loc_index_map = self.gen_index_map(self.coor_df, 'geo_id')
        self.coor_df['geo_id'] = self.coor_df['geo_id'].map(loc_index_map)
        self.df['loc_index'] = self.df['loc_index'].map(loc_index_map)
        user_index_map = self.gen_index_map(self.df, 'user_index')
        self.df['user_index']=self.df['user_index'].map(user_index_map)
        self.num_user=len(user_index_map)
        self.num_loc=len(loc_index_map)
        assert len(loc_index_map) == self.coor_df.shape[0]
        
        for user_index in self.res:
            lens=len(self.res[user_index])
            train_lens=int((1-self.test_scale)*lens)
            # user_index loc_index datetime lat lng timestamp
            for i in range(lens):
                uid=user_index_map[user_index]
                loc_list=[]
                week_list=[]
                timestamp_list=[]
                delta_list=[]
                dist_list=[]
                lats=[]
                longs=[]
                loc_len=len(self.res[user_index][i])
                prev_time=self.res[user_index][i][0][2].timestamp()
                prev_loc=(self.res[user_index][i][0][3],self.res[user_index][i][0][4])
                for row in self.res[user_index][i]:
                    loc_list.append(loc_index_map[row[1]])
                    week_list.append(row[2].weekday())
                    timestamp_list.append(row[5])
                    delta_list.append(row[5]-prev_time)
                    prev_time=row[5]
                    coordist=np.array([row[3]-prev_loc[0],row[4]-prev_loc[1]])
                    dist_list.append(np.sqrt((coordist**2).sum(-1)))
                    prev_loc=[row[3],row[4]]
                    lats.append(row[3])
                    longs.append(row[4])
                if i <= train_lens:
                    self.train_set.append([uid, loc_list, week_list, timestamp_list, loc_len,delta_list,dist_list,lats,longs])
                    self.w2v_data.append([uid, loc_list, week_list, timestamp_list, loc_len])
                else:
                    self.test_set.append([uid, loc_list, week_list, timestamp_list, loc_len,delta_list,dist_list,lats,longs])
        
        self.coor_mat = self.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index').to_numpy()
        self.id2coor_df = self.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index'). \
            set_index('loc_index').sort_index()
        self.user_index_map=user_index_map
        self.loc_index_map=loc_index_map

        # todo list 这里不应该每次都生成，而是应该有缓存，如果有缓存则直接加载
        # 需要保存那些东西呢，train_set, test_set,w2v_set, loc_index_map, user_index_map
        # cache dir 需要包括cut_type，test_scaler, dataset, user_filter, checkin_filter,pre_len
        if self.cache:
            self._logger.info(f'save data cache in {self.cache_file_name}')
            with open(self.cache_file_name,'wb') as f:
                pickle.dump(self.train_set,f)
                pickle.dump(self.test_set,f)
                pickle.dump(self.w2v_data,f)
                pickle.dump(loc_index_map,f)
                pickle.dump(user_index_map,f)
                pickle.dump(self.max_seq_len,f)
                

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {
            "max_seq_len": self.max_seq_len,
            "num_loc": self.num_loc,
            "num_user": self.num_user,
            "train_set": self.train_set,
            "test_set": self.test_set,
            "w2v_data": self.w2v_data,
            "coor_mat": self.coor_mat,
            "id2coor_df": self.id2coor_df,
            "theta" : self.theta,
            "coor_df" : self.coor_df,
            "df":self.df,
            "loc_index_map":self.loc_index_map,
            "offset":self.offset
        }

    def gen_sequence(self, min_len=None, select_days=None, include_delta=False):
        """
        Generate moving sequence from original trajectories.

        @param min_len: minimal length of sentences.
        @param select_day: list of day to select, set to None to use all days.
        """

        if min_len is None:
            min_len = self.min_len
        data = pd.DataFrame(self.df, copy=True)
        data['datetime'] = pd.to_datetime(data["datetime"]) # take long time, can we just store the right format?
        data['day'] = data['datetime'].dt.day
        data['nyr'] = data['datetime'].apply(lambda x: datetime.fromtimestamp(x.timestamp()).strftime("%Y-%m-%d"))
        if select_days is not None:
            data = data[data['nyr'].isin(self.split_days[select_days])]
        
        data['weekday'] = data['datetime'].dt.weekday
        data['timestamp'] = data['datetime'].apply(lambda x: x.timestamp())

        if include_delta:
            data['time_delta'] = data['timestamp'].shift(-1) - data['timestamp']
            coor_delta = (data[['lng', 'lat']].shift(-1) - data[['lng', 'lat']]).to_numpy()
            data['dist'] = np.sqrt((coor_delta ** 2).sum(-1))
        seq_set = []
        for (user_index, day), group in data.groupby(['user_index', 'day']):
            if group.shape[0] < min_len:
                continue
            one_set = [user_index, group['loc_index'].tolist(), group['weekday'].astype(int).tolist(),
                       group['timestamp'].astype(int).tolist(), group.shape[0]]

            if include_delta:
                one_set += [[0] + group['time_delta'].iloc[:-1].tolist(),
                            [0] + group['dist'].iloc[:-1].tolist(),
                            group['lat'].tolist(),
                            group['lng'].tolist()]

            seq_set.append(one_set)
        return seq_set

    def cutter_filter(self):
        """
        切割后的轨迹存储格式: (dict)
            {
                uid: [
                    [
                        checkin_record,
                        checkin_record,
                        ...
                    ],
                    [
                        checkin_record,
                        checkin_record,
                        ...
                    ],
                    ...
                ],
                ...
            }
        """
        # load data according to config
        traj = pd.DataFrame(self.df, copy=True)
        traj['datetime'] = pd.to_datetime(traj["datetime"]) # take long time, can we just store the right format?
        traj['timestamp'] = traj['datetime'].apply(lambda x: x.timestamp())
        # user_set = pd.unique(traj['entity_id'])
        res = {}
        min_session_len = self.min_len # 每个session中至少有3个轨迹
        min_sessions = self.min_sessions # 最少的session数
        window_size = self.time_threshold # 超过24小时就切分,暂时
        cut_method = self.cut_method
        loc_set=set()
        if cut_method == 'time_interval':
            # 按照时间窗口进行切割
            for user_index, group in traj.groupby(['user_index']):
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                lens=group.shape[0]
                
                for index in range(lens):
                    row=group.iloc[index]
                    now_time = row['timestamp']
                    if index == 0:
                        session.append(row.tolist())
                        prev_time = now_time
                    else:
                        time_off = (now_time-prev_time)/3600
                        if time_off < window_size and time_off >= 0 and len(session) < self.max_seq_len:
                            session.append(row.tolist())
                        else:
                            if len(session) >= min_session_len:
                                sessions.append(session)
                            session = []
                            session.append(row.tolist())
                    prev_time = now_time
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    res[user_index] = sessions
        return res

