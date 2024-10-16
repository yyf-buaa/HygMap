import pandas as pd
import ast
from shapely.geometry import LineString
from shapely import wkt
from tqdm import tqdm
df = pd.read_csv("/home/panda/private/jjw/raw_data/pt/regionmap_pt/regionmap_pt.geo")
df['Shape_Leng'] = 0
df['Shape_Area'] = 0
for i in tqdm(range(len(df))):
    polyon = wkt.loads(df.loc[i, 'coordinates'])
    df.loc[i,'Shape_Leng'] = polyon.length
    df.loc[i, 'Shape_Area'] = polyon.area
    df.to_csv("/home/panda/private/jjw/raw_data/pt/regionmap_pt/regionmap_pt.geo")