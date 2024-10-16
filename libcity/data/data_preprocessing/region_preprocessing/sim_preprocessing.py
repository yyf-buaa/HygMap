import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
vectors = np.load('/home/jjw/private/raw_data/sanfransico/region_tf_matrix.npy')
sim_mtx = cosine_similarity(vectors)
non_zero_indices = np.nonzero(sim_mtx)
indices = list(zip(non_zero_indices[0],non_zero_indices[1]))
i = 0
result_df = pd.DataFrame(columns=['rel_id','type','origin_id','destination_id','semantic_weight'])
for indice in tqdm(indices):
    result_df.loc[i] = [i,'sem',indice[0],indice[1],sim_mtx[indice[0],indice[1]]]
    i = i+1
result_df.to_csv('/home/jjw/private/raw_data/sf/regionmap_sf/regionmap_sf.sem')