region_num = 382
road_num = 11095
poi_num = 4056
import pandas as pd
df = pd.read_csv("/home/panda/private/jjw/raw_data/porto/porto.geo")
df_region = df[0:region_num]
df_region = df_region.reset_index(drop=True)
region_columns = ['geo_id','coordinates','region_type','region_name']
df_region = df_region[region_columns]
df_region = df_region.rename({'region_type':'type','region_name':'name'})

df_road = df[region_num:region_num+road_num]
df_road = df_road.reset_index(drop=True)
df_road['geo_id'] = df_road.index.values
road_columns = ['geo_id','coordinates','road_highway','road_lanes','road_length','road_maxspeed']
df_road = df_road[road_columns]
df_road = df_road.rename({'road_highway':'highway','road_lanes':'lanes','road_length':'length','road_maxspeed':'maxspeed'})

df_poi = df[region_num+road_num:]
df_poi = df_poi.reset_index(drop=True)
df_poi['geo_id'] = df_poi.index.values
poi_columns = ['geo_id','coordinates','poi_type']
df_poi = df_poi[poi_columns]
df_poi = df_poi.rename({'poi_type':'type'})
df_region.to_csv("/home/panda/private/jjw/raw_data/pt/regionmap_pt/regionmap.geo")
df_road.to_csv("/home/panda/private/jjw/raw_data/pt/roadmap_pt/roadmap.geo")
df_poi.to_csv("/home/panda/private/jjw/raw_data/pt/poimap_pt/poimap.geo")