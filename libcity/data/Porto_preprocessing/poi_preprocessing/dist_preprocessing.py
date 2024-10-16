region_num = 382
road_num = 11095
poi_num = 4056
import numpy as np
import pandas as pd
from tqdm import tqdm
dist_mtx = np.load("/home/panda/private/jjw/raw_data/pt/poimap_pt/poimap_pt_dist_mtx.npy")
threshold = dist_mtx.mean()/10
weight = []
dist_mtx += np.eye(poi_num)
indices = np.where(dist_mtx<threshold)
index_pairs = list(zip(indices[0], indices[1]))
cnt = 0
with open('/home/panda/private/jjw/raw_data/pt/poimap_pt/poimap_pt.rel','w') as file:
    file.write('{},{},{},{},{}\n'.format('rel_id','type','origin_id','destination_id','geographical_dist','geographical_weight'))
    for indice in tqdm(index_pairs):
        i = indice[0]
        j = indice[1]
        file.write('{},{},{},{},{}\n'.format(cnt,'rel',i,j,dist_mtx[i][j],0))
        weight.append(1 / (dist_mtx[i][j]+threshold/10))
        cnt+=1
weight_min = min(weight)
weight_max = max(weight)
normalized_weight = [(x - weight_min) / (weight_max - weight_min) for x in weight]
rel_df = pd.read_csv('/home/panda/private/jjw/raw_data/pt/poimap_pt/poimap_pt.rel')
rel_df['geographical_weight'] = normalized_weight
rel_df.to_csv('/home/panda/private/jjw/raw_data/pt/poimap_pt/poimap_pt.rel')