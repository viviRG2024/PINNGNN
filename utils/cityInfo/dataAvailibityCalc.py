import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

import sys
sys.path.append(r'..')
from utils.utils import load_config
if __name__ == "__main__":
    config = load_config(r'config.json')
    cities = config['cities']

    results = []

    base_folder = 'data\input\cities'
    for city in cities:
        city_path = os.path.join(base_folder, city)    
        if os.path.isdir(city_path):
            # read {city_name}_sensor.csv
            sensor_csv_path = os.path.join(city_path, f'detectors_public.csv')
            detector_csv_path = os.path.join(city_path, f'detectors_public.csv')
            if os.path.exists(sensor_csv_path):
                sensor_data = pd.read_csv(sensor_csv_path)
                detector = pd.read_csv(detector_csv_path)
                sensor_total = sensor_data.shape[0] # the number of sensors
                
                # calculate bounding box
                min_lon, min_lat = detector['long'].min(), detector['lat'].min()
                max_lon, max_lat = detector['long'].max(), detector['lat'].max()
                bounding_box = box(min_lon, min_lat, max_lon, max_lat)
            else:
                sensor_total = 0
                bounding_box = None

            # read roads.gpkg
            roads_gpkg_path = os.path.join(city_path, 'roads.gpkg')
            if os.path.exists(roads_gpkg_path):
                roads_data = gpd.read_file(roads_gpkg_path)

                # filter roads by bounding box
                if bounding_box:
                    roads_data = roads_data[roads_data.intersects(bounding_box)]

                line_string_total = roads_data.shape[0]  # the number of LineString
            else:
                line_string_total = 0

            # calculate data availability
            if line_string_total != 0:
                data_availability = round(sensor_total / line_string_total,4)
            else:
                data_availability = 0

            # add result to results
            results.append({
                'City': city,
                'Sensor Total': sensor_total,
                'LineString Total': line_string_total,
                'Data Availability': data_availability
            })

    results_df = pd.DataFrame(results)
    # normalize data availability
    max_data_availability = results_df['Data Availability'].max()
    if max_data_availability != 0:
        results_df['Normalized Data Availability'] = round(results_df['Data Availability'] / max_data_availability,4)
    else:
        results_df['Normalized Data Availability'] = 0

    # save results to city_data_availability.csv
    results_df.to_csv(os.path.join(r'data\output','city_data_availability.csv'), index=False)

    print("Results have been saved to city_data_availability.csv")
