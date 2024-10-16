region_num = 382
road_num = 11095
poi_num = 4056
import numpy as np
trans_mtx = np.zeros(shape=[region_num,region_num])
import pandas as pd
from tqdm import  tqdm
traj_df = pd.read_csv("/home/panda/private/jjw/raw_data/porto/porto.gpstraj")
rel_df = pd.read_csv("/home/panda/private/jjw/raw_data/porto/porto.rel")
rel_df = rel_df[rel_df['rel_type'] == 'road2region'].reset_index(drop=True)
rel_dic = {}
from tqdm import  tqdm
for i in tqdm(range((len(rel_df)))):
    rel_dic[rel_df.loc[i,'origin_id']] = rel_df.loc[i,'destination_id']
for i in tqdm(range(len(traj_df)-1)):
    if traj_df.loc[i,'traj_id'] == traj_df.loc[i+1,'traj_id']:
        origin_road_id = int(traj_df.loc[i,'location'])
        destin_road_id = int(traj_df.loc[i+1, 'location'])
        origin_id = rel_dic[origin_road_id]
        destin_id = rel_dic[destin_road_id]
        trans_mtx[origin_id,destin_id]+=1
# row_sums = trans_mtx.sum(axis=1) + 0.0000001
# trans_mtx = trans_mtx / row_sums[:,np.newaxis]
min_val = np.min(trans_mtx)
max_val = np.max(trans_mtx)
trans_mtx = (trans_mtx - min_val) / (max_val - min_val)
non_zero_indices = np.nonzero(trans_mtx)
indices = list(zip(non_zero_indices[0],non_zero_indices[1]))
i = 0
result_df = pd.DataFrame(columns=['rel_id','type','origin_id','destination_id','mobility_weight'])
for indice in indices:
    result_df.loc[i] = [i,'mob',indice[0],indice[1],trans_mtx[indice[0],indice[1]]]
    i = i+1
result_df.to_csv("/home/panda/private/jjw/raw_data/pt/regionmap_pt/regionmap_pt.mob")