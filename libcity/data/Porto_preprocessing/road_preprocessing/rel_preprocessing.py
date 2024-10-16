#根据处理什么类型的实体来改一下路径和rel_type
import pandas as pd
import ast
from shapely.geometry import LineString
from tqdm import tqdm
region_num = 382
road_num = 11095
poi_num = 4056
rel_df = pd.read_csv("/home/panda/private/jjw/raw_data/porto/porto.rel")
rel_df = rel_df[rel_df['rel_type']=='road2road']
rel_df = rel_df.reset_index(drop=True)
rel_df = rel_df.drop(columns=['rel_type'])
rel_df['geographical_dist'] = 0
#计算两条路之间的dist
geo_df = pd.read_csv("/home/panda/private/jjw/raw_data/pt/roadmap_pt/roadmap_pt.geo")
weight = []
max_weight = 0
for i in tqdm(range(len(rel_df))):
    origin_id = rel_df.loc[i,'origin_id']-region_num
    destination_id = rel_df.loc[i,'destination_id']-region_num
    origin_coor_str = geo_df[geo_df['geo_id'] == origin_id]['coordinates'].values[0]
    destin_coor_str = geo_df[geo_df['geo_id'] == destination_id]['coordinates'].values[0]
    origin_road = LineString(ast.literal_eval(origin_coor_str))
    destin_road = LineString(ast.literal_eval(destin_coor_str))
    distance = origin_road.centroid.distance(destin_road.centroid)
    rel_df.loc[i,'geographical_dist'] = distance
    if distance == 0:
        weight.append(float('inf'))
    else:
        weight.append(1/(distance))
        if 1/distance > max_weight:
            max_weight = 1/distance
weight = [max_weight if x == float('inf') else x for x in weight]
weight_min = min(weight)
weight_max = max(weight)
normalized_weight = [(x - weight_min) / (weight_max - weight_min) for x in weight]
rel_df['geographical_weight'] = normalized_weight
rel_df.to_csv("/home/panda/private/jjw/raw_data/pt/roadmap_pt/roadmap_pt.rel")
