region_num = 382
road_num = 11095
poi_num = 4056
import numpy as np
checkin = np.zeros(shape=[poi_num])
import pandas as pd
traj_df = pd.read_csv("/home/panda/private/tangyb/tyb/remote/representation/raw_data/new/embeddings/new0909/sanfransico_poi_test.csv")
from tqdm import tqdm
for i in tqdm(range(len(traj_df))):
    location = traj_df.loc[i,'location']
    checkin[location]+=1
np.save("/home/panda/private/jjw/yyf/HOME-GCL-main/raw_data/sf/poi_check_in.npy",checkin)
