import pandas as pd
import ast
from shapely.geometry import LineString
from tqdm import tqdm
df = pd.read_csv("/home/panda/private/jjw/raw_data/sf/roadmap_sf/roadmap_sf.geo")
for i in tqdm(range(len(df))):
    coor_str = df.loc[i,'coordinates']
    road = LineString(ast.literal_eval(coor_str))
    df.loc[i,'m_lon'] = road.centroid.x
    df.loc[i, 'm_lat'] = road.centroid.y
df.to_csv("/home/panda/private/jjw/raw_data/sf/roadmap_sf/roadmap_sf.geo")