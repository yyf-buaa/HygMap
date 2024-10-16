road_num = 27274
region_num = 194
poi_num = 15674
import pandas as pd
import numpy as np
df = pd.read_csv("/home/panda/private/jjw/raw_data/sanfransico/sanfransico.od")
od_flow = np.zeros(shape=[road_num,road_num])
from tqdm import tqdm
for i in tqdm(range(len(df))):
    import ipdb
    ipdb.set_trace()
    origin_road = df.loc[i,'origin_id']-road_num
    destin_road = df.loc[i,'destination_id']-road_num
    od_flow[origin_road][destin_road]+=df.loc[i,'flow']
np.save('/home/panda/private/jjw/yyf/HOME-GCL-main/raw_data/sf/road_od_flow_sf.npy',od_flow)