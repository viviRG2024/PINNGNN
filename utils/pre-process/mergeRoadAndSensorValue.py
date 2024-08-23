import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import json
import time

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def round_coordinates(coords, digits=3):
    return tuple(round(coord, digits) for coord in coords)

def gdf2graph(gdf):
    G = nx.Graph()
    for _, row in gdf.iterrows():
        start_point = round_coordinates(row['geometry'].coords[0])
        end_point = round_coordinates(row['geometry'].coords[-1])
        
        edge_attributes = {key: value for key, value in row.items() if key != 'geometry'}
        G.add_edge(start_point, end_point, **edge_attributes)
    
    return G

def nodegraph2edgegraph(G):
    H = nx.Graph()

    for u, v, data in G.edges(data=True):
        road_id = data['road_id']
        if not H.has_node(road_id):
            H.add_node(road_id, **data)

    for node in G.nodes():
        connected_edges = list(G.edges(node, data=True))
        for i in range(len(connected_edges) - 1):
            for j in range(i + 1, len(connected_edges)):
                edge_i_id = connected_edges[i][2]['road_id']
                edge_j_id = connected_edges[j][2]['road_id']
                if not H.has_edge(edge_i_id, edge_j_id):
                    H.add_edge(edge_i_id, edge_j_id, distance=(connected_edges[i][2]['road_length'] + connected_edges[j][2]['road_length']) / 2)

    return H

def process_city_data(city_name, base_folder):
    road_file = os.path.join(base_folder, 'selected_network_4326.geojson')
    sensor_file = os.path.join(base_folder, f'{city_name}_sensor.csv')

    if not os.path.exists(road_file) or not os.path.exists(sensor_file):
        print(f"Files do not exist for city {city_name}")
        return

    # read data
    road_data = gpd.read_file(road_file, driver='GeoJSON')
    sensor_data = pd.read_csv(sensor_file)

    # construct graph
    G = gdf2graph(road_data)
    H = nodegraph2edgegraph(G)

    # create distance data(from, to, distance)
    distance_data = []
    for u, v, data in H.edges(data=True):               
        distance_data.append({
            'from': H.nodes[u]['road_id'], 
            'to': H.nodes[v]['road_id'],  
            'distance': data['distance']
        })
    
    distance_df = pd.DataFrame(distance_data)
    distance_df.to_csv(os.path.join(base_folder, f'{city_name}_distance_hours.csv'), index=False)
    print(f"Processed distance data for {city_name}")
# region implementate by 5mins
    # create npz data
    # sensor_data['datetime'] = pd.to_datetime(sensor_data['day']) + pd.to_timedelta(sensor_data['interval'], unit='s')
    # start_date = sensor_data['datetime'].min().floor('D')
    # end_date = sensor_data['datetime'].max().ceil('D')
    # date_range = pd.date_range(start=start_date, end=end_date, freq='5T')
    
    # num_timestamps = len(date_range)
    # all_road_ids = road_data['road_id'].unique()
    # num_roads = len(all_road_ids)
    
    # data_array = np.full((num_roads, num_timestamps, 3), -1, dtype=float)  # default to -1
    # print(data_array.shape)
    
    # road_id_to_index = {road_id: idx for idx, road_id in enumerate(all_road_ids)}
    # road_id_to_detid = dict(zip(road_data['road_id'], road_data['detid']))

    # for _, row in sensor_data.iterrows():
    #     detid = row['detid']
    #     road_ids = [road_id for road_id, det_id in road_id_to_detid.items() if det_id == detid]
    #     for road_id in road_ids:
    #         if road_id in road_id_to_index:
    #             road_idx = road_id_to_index[road_id]
    #             time_idx = date_range.get_loc(row['datetime'], method='nearest')
    #             data_array[road_idx, time_idx, 0] = row['flow'] if not pd.isna(row['occ']) else -1
    #             data_array[road_idx, time_idx, 1] = row['occ'] if not pd.isna(row['occ']) else -1
    #             data_array[road_idx, time_idx, 2] = row['speed'] if not pd.isna(row['speed']) else -1

    # np.savez_compressed(os.path.join(base_folder, f'{city_name}_data_hours.npz'), data=data_array)
    # print(f"Processed npz data for {city_name}")

    # print(f"Processed data for {city_name}")
    # print(f"Data shape: {data_array.shape}")
# endregion

# region implementate by 1hour
    # deal with sensor data, merge with road data
    sensor_data['datetime'] = pd.to_datetime(sensor_data['day']) + pd.to_timedelta(sensor_data['interval'], unit='s')
    sensor_data['hour'] = sensor_data['datetime'].dt.floor('H')

    # group by detid and hour
    hourly_data = sensor_data.groupby(['detid', 'hour']).agg({
        'flow': 'mean',
        'occ': 'mean',
        'speed': 'mean'
    }).reset_index()

    start_date = hourly_data['hour'].min()
    end_date = hourly_data['hour'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    num_timestamps = len(date_range)
    all_road_ids = road_data['road_id'].unique()
    num_roads = len(all_road_ids)
    
    data_array = np.full((num_roads, num_timestamps, 3), -1, dtype=float)  # default to -1
    print(data_array.shape)
    
    road_id_to_index = {road_id: idx for idx, road_id in enumerate(all_road_ids)}
    road_id_to_detid = dict(zip(road_data['road_id'], road_data['detid']))

    npz_path=os.path.join(base_folder, f'{city}_data_hours.npz')
    if not os.path.exists(npz_path):
        for _, row in hourly_data.iterrows():
            detid = row['detid']
            road_ids = [road_id for road_id, det_id in road_id_to_detid.items() if det_id == detid]
            for road_id in road_ids:
                if road_id in road_id_to_index:
                    road_idx = road_id_to_index[road_id]
                    time_idx = date_range.get_loc(row['hour'])
                    data_array[road_idx, time_idx, 0] = row['flow']
                    data_array[road_idx, time_idx, 1] = row['occ']
                    data_array[road_idx, time_idx, 2] = row['speed'] if not pd.isna(row['speed']) else -1
        np.savez_compressed(npz_path, data=data_array)
    else:
        print(f"File already exists: {npz_path}")
    print(f"Processed data for {city}")
    print(f"Data shape: {data_array.shape}")
# endregion

if __name__ == '__main__':
    t_start = time.time()
    
    config = load_config(r'config.json')
    cities = config['cities']
    base_folder = r'data/input/cities'

    for city in cities[6:7]:
        city_dir= os.path.join(base_folder, city)
        print(f"Processing data for {city}")
        process_city_data(city, city_dir)
        
    print('Time cost: %.2f s' % (time.time() - t_start))