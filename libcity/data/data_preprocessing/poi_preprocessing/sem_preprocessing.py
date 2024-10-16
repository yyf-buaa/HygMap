road_num = 27274
region_num = 194
poi_num = 15674
#要做成一个包含有road_num个文档的文档集
import pandas as pd
import numpy as np

from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# 初始化 BERT tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
model = BertModel.from_pretrained('google-bert/bert-base-uncased')
from tqdm import tqdm
def encode_sentences(sentences):
    # 编码每个句子
    encoded_layers = []
    for sentence in tqdm(sentences):
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=10)
        with torch.no_grad():
            outputs = model(**inputs)
        # 获取 [CLS] token 的 embedding 作为句子的表示
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        encoded_layers.append(cls_embedding)

    # 将结果堆叠成一个矩阵
    return np.vstack(encoded_layers)

poi_df = pd.read_csv("/home/jjw/private/raw_data/sanfransico/POI_sanfransico.csv")
poi_type_list = poi_df['3'].values
sem_mtx = encode_sentences(poi_type_list)
np.save('/home/jjw/private/raw_data/sf/poimap_sf/sem_mtx.npy',sem_mtx)
