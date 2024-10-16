import pandas as pd
import ast
from shapely.geometry import LineString
from shapely import wkt
from tqdm import tqdm
df = pd.read_csv('~/private/jjw/raw_data/sf/regionmap_sf/regionmap_sf.geo')
df['m_lon'] = 0
df['m_lat'] = 0
for i in tqdm(range(len(df))):
    polyon = wkt.loads(df.loc[i, 'geometry'])
    df.loc[i,'m_lon'] = polyon.centroid.x
    df.loc[i, 'm_lat'] = polyon.centroid.y
df.to_csv('~/private/jjw/raw_data/sf/regionmap_sf/regionmap_sf.geo')