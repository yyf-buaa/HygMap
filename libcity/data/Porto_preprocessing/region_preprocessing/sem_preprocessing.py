# region_num = 382
# road_num = 11095
# poi_num = 4056
# #要做成一个包含有road_num个文档的文档集
# import pandas as pd
# import numpy as np
# poi_df = pd.read_csv("/home/panda/private/jjw/raw_data/pt/poimap_pt/poimap_pt.geo")
# poi_type_list = poi_df['poi_type'].unique().tolist()
# documents = []
# rel_df = pd.read_csv("/home/panda/private/jjw/raw_data/porto/porto.rel")
# from tqdm import tqdm
# for i in tqdm(range(region_num)):
#     result_str = ''
#     poi_list = rel_df[(rel_df['rel_type'] == 'region2poi') & (rel_df['origin_id'] == i)]['destination_id'].values
#     for poi in poi_list:
#         poi_id = poi-road_num-region_num
#         poi_type = poi_type_list.index(poi_df.loc[poi_id,'poi_type'])
#         result_str += 'poi_{} '.format(poi_type)
#     documents.append(result_str)
# file_path = '/home/panda/private/jjw/raw_data/porto/region_poi_doc.txt'
# with open(file_path,'w') as file:
#     for line in documents:
#         file.write(line+'\n')
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(documents)
# dense_tf_matrix = tfidf_matrix.todense()
# np.save('/home/panda/private/jjw/raw_data/porto/region_tf_matrix.npy',dense_tf_matrix)
#
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
vectors = np.load('/home/panda/private/jjw/raw_data/porto/region_tf_matrix.npy')
sim_mtx = cosine_similarity(vectors)
non_zero_indices = np.nonzero(sim_mtx)
indices = list(zip(non_zero_indices[0],non_zero_indices[1]))
i = 0
result_df = pd.DataFrame(columns=['rel_id','type','origin_id','destination_id','semantic_weight'])
for indice in tqdm(indices):
    result_df.loc[i] = [i,'sem',indice[0],indice[1],sim_mtx[indice[0],indice[1]]]
    i = i+1
result_df.to_csv("/home/panda/private/jjw/raw_data/pt/regionmap_pt/regionmap_pt.sem")