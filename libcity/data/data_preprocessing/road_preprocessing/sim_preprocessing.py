import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
vectors = np.load('/home/jjw/private/raw_data/sanfransico/tf_matrix.npy')
sim_mtx = cosine_similarity(vectors)
threshold = 0.5
indices = np.where(sim_mtx>threshold)
indices = list(zip(indices[0],indices[1]))
i = 0
with open('/home/jjw/private/raw_data/sf/roadmap_sf/roadmap_sf.sem','w') as file:
    file.write('{},{},{},{},{}\n'.format('rel_id', 'type', 'origin_id', 'destination_id', 'semantic_weight'))
    for indice in tqdm(indices):
        file.write('{},{},{},{},{}\n'.format(i,'sem',indice[0],indice[1],sim_mtx[indice[0],indice[1]]))
        i = i+1