road_num = 27274
region_num = 194
poi_num = 15674
import pandas as pd
import numpy as np
df = pd.read_csv("/home/panda/private/jjw/raw_data/sanfransico/sanfransico.od")
od_flow = np.zeros(shape=[region_num,region_num])
from tqdm import  tqdm
for i in tqdm(range(len(df))):
    origin_region = df.loc[i,'origin_id']
    destin_region = df.loc[i,'destination_id']
    od_flow[origin_region][destin_region]+=df.loc[i,'flow']
np.save('/home/panda/private/jjw/yyf/HOME-GCL-main/raw_data/sf/region_od_flow_sf.npy',od_flow)