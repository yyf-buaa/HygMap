import json
import pandas as pd
region_num = 194
df = pd.read_csv('/home/panda/private/jjw/raw_data/sanfransico/sanfransico.rel')
df = df[(df['rel_type'] == 'region2road') | (df['rel_type'] == 'region2poi')].reset_index(drop=True)
dict = {}
for i in range(region_num):
    dict[i] = []
for i in range(len(df)):
    origin_id = df.loc[i,'origin_id']
    destin_id = df.loc[i,'destination_id']
    dict[origin_id].append(destin_id)
import ipdb
ipdb.set_trace()
with open('/home/panda/private/jjw/raw_data/sf/region2road_poi.json', 'w') as file:
    json.dump(dict, file, indent=4)