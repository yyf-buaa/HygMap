region_num = 382
road_num = 11095
poi_num = 4056
import numpy as np
od_mtx = np.load("/home/panda/private/jjw/raw_data/porto/traj_road_train_od.npy")
min_val = np.min(od_mtx)
max_val = np.max(od_mtx)
import pandas as pd
# 归一化矩阵
normalized_matrix = (od_mtx - min_val) / (max_val - min_val)
non_zero_indices = np.nonzero(normalized_matrix)
indices = list(zip(non_zero_indices[0],non_zero_indices[1]))
i = 0
from tqdm import tqdm
result_df = pd.DataFrame(columns=['rel_id','type','origin_id','destination_id','mobility_weight'])
for indice in tqdm(indices):
    result_df.loc[i] = [i,'mob',indice[0],indice[1],normalized_matrix[indice[0],indice[1]]]
    i = i+1
result_df.to_csv('/home/panda/private/jjw/raw_data/pt/roadmap_pt/roadmap_pt.mob')

od_mtx = np.load("/home/panda/private/jjw/raw_data/porto/traj_region_train_od.npy")
min_val = np.min(od_mtx)
max_val = np.max(od_mtx)
import pandas as pd
# 归一化矩阵
normalized_matrix = (od_mtx - min_val) / (max_val - min_val)
non_zero_indices = np.nonzero(normalized_matrix)
indices = list(zip(non_zero_indices[0],non_zero_indices[1]))
i = 0
from tqdm import tqdm
result_df = pd.DataFrame(columns=['rel_id','type','origin_id','destination_id','mobility_weight'])
for indice in tqdm(indices):
    result_df.loc[i] = [i,'mob',indice[0],indice[1],normalized_matrix[indice[0],indice[1]]]
    i = i+1
result_df.to_csv('/home/panda/private/jjw/raw_data/pt/regionmap_pt/regionmap_pt.mob')