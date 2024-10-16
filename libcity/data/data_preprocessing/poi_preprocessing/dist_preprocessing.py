road_num = 27274
region_num = 194
poi_num = 15674
import numpy as np
import pandas as pd
from tqdm import tqdm
dist_mtx = np.load("/home/jjw/private/raw_data/sf/poimap_sf/poimap_sf_dist_mtx.npy")
threshold = dist_mtx.mean()/10
weight = []
dist_mtx += np.eye(poi_num)
indices = np.where(dist_mtx<threshold)
index_pairs = list(zip(indices[0], indices[1]))
cnt = 0
with open('/home/jjw/private/raw_data/sf/poimap_sf/poimap_sf.rel','w') as file:
    file.write('{},{},{},{},{}\n'.format('rel_id','type','origin_id','destination_id','geographical_dist','geographical_weight'))
    for indice in tqdm(index_pairs):
        i = indice[0]
        j = indice[1]
        file.write('{},{},{},{},{}\n'.format(cnt,'rel',i,j,dist_mtx[i][j],0))
        weight.append(1 / (dist_mtx[i][j]+0.00001))
        cnt+=1
weight_min = min(weight)
weight_max = max(weight)
normalized_weight = [(x - weight_min) / (weight_max - weight_min) for x in weight]
rel_df = pd.read_csv('/home/jjw/private/raw_data/sf/poimap_sf/poimap_sf.rel')
rel_df['geographical_weight'] = normalized_weight
rel_df.to_csv('/home/jjw/private/raw_data/sf/poimap_sf/poimap_sf.rel')