#根据处理什么类型的实体来改一下路径和rel_type
import pandas as pd
import ast
from shapely.geometry import LineString
from shapely import wkt
from tqdm import tqdm
rel_df = pd.DataFrame(columns=['rel_id','type','origin_id','destination_id','geographical_dist','geographical_weight'])
#计算两条路之间的dist
geo_df = pd.read_csv('/home/panda/private/yangyifan/yyf/raw_data/sanfransico/region_sanfransico.csv')
weight = []
polyons = []
for i in tqdm(range(len(geo_df))):
    polyons.append(wkt.loads(geo_df.loc[i,'geometry']))
for i in tqdm(range(len(geo_df))):
    for j in range(len(geo_df)):
        if (not i == j) and (polyons[i].touches(polyons[j])):
            distance = polyons[i].centroid.distance(polyons[j].centroid)
            rel_df.loc[len(rel_df)] = [len(rel_df),'rel',i,j,distance,0]
            weight.append(1 / (distance))
    # origin_id = rel_df.loc[i,'origin_id']-194
    # destination_id = rel_df.loc[i,'destination_id']-194
    # origin_coor_str = geo_df[geo_df['geo_id'] == origin_id]['coordinates'].values[0]
    # destin_coor_str = geo_df[geo_df['geo_id'] == destination_id]['coordinates'].values[0]
    # origin_road = LineString(ast.literal_eval(origin_coor_str))
    # destin_road = LineString(ast.literal_eval(destin_coor_str))
    # distance = origin_road.distance(destin_road)
    # rel_df.loc[i,'geographical_dist'] = distance
    # weight.append(1/(distance+0.0001))

weight_min = min(weight)
weight_max = max(weight)
normalized_weight = [(x - weight_min) / (weight_max - weight_min) for x in weight]
rel_df['geographical_weight'] = normalized_weight
rel_df.to_csv('/home/jjw/private/raw_data/sf/regionmap_sf/regionmap_sf.rel')
