road_num = 27274
region_num = 194
poi_num = 15674
import numpy as np
trans_mtx = np.zeros(shape=[region_num,region_num])
import pandas as pd
from tqdm import  tqdm
traj_df = pd.read_csv("/home/panda/private/jjw/yyf/HOME-GCL-main/raw_data/sf/traj_region_train.csv")
for i in tqdm(range(len(traj_df))):
    if traj_df.loc[i,'traj_id'] == traj_df.loc[i+1,'traj_id']:
        origin_road_id = int(traj_df.loc[i,'geo_id'])
        destin_road_id = int(traj_df.loc[i+1, 'geo_id'])
        origin_id = rel_dic[origin_road_id]
        destin_id = rel_dic[destin_road_id]
        trans_mtx[origin_id,destin_id]+=1
row_sums = trans_mtx.sum(axis=1) + 0.0000001
trans_mtx = trans_mtx / row_sums[:,np.newaxis]
non_zero_indices = np.nonzero(trans_mtx)
indices = list(zip(non_zero_indices[0],non_zero_indices[1]))
i = 0
result_df = pd.DataFrame(columns=['rel_id','type','origin_id','destination_id','mobility_weight'])
for indice in indices:
    result_df.loc[i] = [i,'mob',indice[0],indice[1],trans_mtx[indice[0],indice[1]]]
    i = i+1
result_df.to_csv('/home/jjw/private/raw_data/sf/regionmap_sf/regionmap_sf.mob')