import pandas as pd
import numpy as np

df=pd.read_csv("/home/panda/private/yangyifan/dm24_dataset/traj_nexthop.csv")
next_traj_id = 0
for i in range(len(df)-1):
    traj_id = df.loc[i,'trajectory_id']
    next_traj_id = df.loc[i+1,'trajectory_id']
    if not traj_id == next_traj_id:
        df.loc[i,'coordinates'] = np.nan
df.loc[len(df)-1,'coordinates'] = np.nan
df.to_csv("/home/panda/private/yangyifan/dm24_dataset/traj_next_without_label.csv")
