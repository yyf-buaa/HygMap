import pandas as pd
df = pd.read_csv("/home/panda/private/jjw/raw_data/porto/porto.geo")
df_region = df[df['type'] == 'Polygon'].reset_index(drop=True)
df_region = df_region['geo_id','region_type','region_name','region_geometry']
df_region = df_region.rename(columns={'region_type':'type','region_name':'name','region_geometry':'geometry'})