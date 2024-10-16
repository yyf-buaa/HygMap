import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
vectors = np.load('/home/panda/private/jjw/raw_data/pt/poimap_pt/sem_mtx.npy')
sim_mtx = cosine_similarity(vectors)
threshold = 0.97
indices = np.where(sim_mtx>threshold)
indices = list(zip(indices[0],indices[1]))
i = 0
with open('/home/panda/private/jjw/raw_data/pt/poimap_pt/poimap_pt.sem','w') as file:
    file.write('{},{},{},{},{}\n'.format('rel_id','type','origin_id','destination_id','semantic_weight'))
    for indice in tqdm(indices):
        # if i == 100:
        #     break
        file.write('{},{},{},{},{}\n'.format(i,'sem',indice[0],indice[1],(sim_mtx[indice[0],indice[1]]-threshold)/(1-threshold)))
        i = i+1