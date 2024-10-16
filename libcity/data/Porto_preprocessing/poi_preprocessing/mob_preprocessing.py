region_num = 382
road_num = 11095
poi_num = 4056
import numpy as np
trans_mtx = np.zeros(shape=[poi_num,poi_num])
import pandas as pd
traj_df = pd.read_csv("/home/panda/private/jjw/raw_data/porto/porto.poitraj")
from tqdm import  tqdm
user_num = 578
for user in tqdm(range(user_num)):
    user_traj = traj_df[traj_df['entity_id']==user].reset_index(drop=True)
    for i in range(len(user_traj)-1):
        origin_id = user_traj.loc[i,'location']
        destin_id = user_traj.loc[i+1,'location']
        trans_mtx[origin_id, destin_id] += 1
row_sums = trans_mtx.sum(axis=1) + 0.0000001
trans_mtx = trans_mtx / row_sums[:,np.newaxis]
non_zero_indices = np.nonzero(trans_mtx)
indices = list(zip(non_zero_indices[0],non_zero_indices[1]))
i = 0
result_df = pd.DataFrame(columns=['rel_id','type','origin_id','destination_id','mobility_weight'])
for indice in tqdm(indices):
    result_df.loc[i] = [i,'mob',indice[0],indice[1],trans_mtx[indice[0],indice[1]]]
    i = i+1
result_df.to_csv('/home/panda/private/jjw/raw_data/pt/poimap_pt/poimap_pt.mob')