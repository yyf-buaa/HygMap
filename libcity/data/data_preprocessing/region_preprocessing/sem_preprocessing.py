road_num = 27274
region_num = 194
poi_num = 15674
#要做成一个包含有road_num个文档的文档集
import pandas as pd
import numpy as np
poi_df = pd.read_csv("/home/jjw/private/raw_data/sanfransico/POI_sanfransico.csv")
poi_type_list = poi_df['3'].unique().tolist()
documents = []
rel_df = pd.read_csv("/home/jjw/private/raw_data/sanfransico/sanfransico.rel")
from tqdm import tqdm
for i in tqdm(range(region_num)):
    result_str = ''
    poi_list = rel_df[(rel_df['rel_type'] == 'region2poi') & (rel_df['origin_id'] == i)]['destination_id'].values
    for poi in poi_list:
        poi_id = poi-road_num-region_num
        poi_type = poi_type_list.index(poi_df.loc[poi_id,'3'])
        result_str += 'poi_{} '.format(poi_type)
    documents.append(result_str)
file_path = '/home/jjw/private/raw_data/sanfransico/region_poi_doc.txt'
with open(file_path,'w') as file:
    for line in documents:
        file.write(line+'\n')
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
dense_tf_matrix = tfidf_matrix.todense()
np.save('/home/jjw/private/raw_data/sanfransico/region_tf_matrix.npy',dense_tf_matrix)

