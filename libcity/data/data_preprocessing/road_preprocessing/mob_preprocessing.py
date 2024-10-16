road_num = 27274
import numpy as np
trans_mtx = np.zeros(shape=[road_num,road_num])
import pandas as pd
traj_df = pd.read_csv("/home/panda/private/jjw/raw_data/sanfransico/sanfransico.gpstraj")
# rel_df = pd.read_csv("/home/panda/private/jjw/raw_data/sanfransico/sanfransico.rel")
# rel_df = rel_df[rel_df['rel_type'] == 'road2region'].reset_index(drop=True)
# rel_dic = {}
from tqdm import  tqdm
# for i in tqdm(range((len(rel_df)))):
#     rel_dic[rel_df.loc[i,'origin_id']] = rel_df.loc[i,'destination_id']
traj_list = []
od_mtx = np.zeros(shape=[road_num,road_num])
for i in tqdm(range(len(traj_df)-1)):
    if traj_df.loc[i,'traj_id'] == traj_df.loc[i+1,'traj_id']:
        origin_id = int(traj_df.loc[i,'geo_id'])-194
        destin_id = int(traj_df.loc[i+1, 'geo_id'])-194
        # origin_id = rel_dic[origin_id]
        # destin_id = rel_dic[destin_id]
        trans_mtx[origin_id,destin_id]+=1
        traj_list.append(origin_id)
    else:
        traj_list.append(int(traj_df.loc[i,'geo_id'])-194)
        od_mtx[traj_list[0]][traj_list[-1]]+=1
        traj_list = []
traj_list.append(int(traj_df.loc[len(traj_df)-1,'geo_id'])-194)
od_mtx[traj_list[0]][traj_list[-1]]+=1
outflow = np.sum(trans_mtx, axis = 0)/30.0
inflow = np.sum(trans_mtx, axis = 1)/30.0
np.save('/home/panda/private/jjw/yyf/HOME-GCL-main/raw_data/sf/road_od_flow.npy',od_mtx)
np.save('/home/panda/private/jjw/yyf/HOME-GCL-main/raw_data/sf/road_in_flow.npy',inflow)
np.save('/home/panda/private/jjw/yyf/HOME-GCL-main/raw_data/sf/road_out_flow.npy',outflow)