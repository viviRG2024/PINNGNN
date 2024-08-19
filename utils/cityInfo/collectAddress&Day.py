import os
import pandas as pd
import requests
from geopy.geocoders import Nominatim

def get_city_coordinates(city_name):
    location = geolocator.geocode(city_name)
    if location:
        return (location.latitude, location.longitude)
    else:
        return None

if __name__ == '__main__':
    in_dir = r'data\output\city'
    out_dir=r'data\output'

    # intialize geolocator
    geolocator = Nominatim(user_agent="pinns-gnn-weather-download-agent")

    all_data = []

    for city_folder in os.listdir(in_dir):
        city_folder_path = os.path.join(in_dir, city_folder)
        
        if os.path.isdir(city_folder_path):
            sensor_file_path = os.path.join(city_folder_path, f'{city_folder}_sensor.csv')
            
            if os.path.exists(sensor_file_path):
                df = pd.read_csv(sensor_file_path)
                if 'day' in df.columns:
                    unique_days = df['day'].unique()
                    coordinates = get_city_coordinates(city_folder)
                    if not coordinates:
                        print(f"cannot get {city_folder} coordinates")
                        continue
                    
                    city_lat, city_lon = coordinates
                    print(f"city:{city_folder},lat:{city_lat},lon:{city_lon}, days:{len(unique_days)}")
                    # print(unique_days)
                    
                    # add city data to all_data
                    for date in unique_days:
                        all_data.append({'city': city_folder, 'lat':city_lat,'lon':city_lon,'date': date})
                else:
                    print(f"{sensor_file_path} file does not contain 'day' column")
            else:
                print(f"{sensor_file_path} does not exist")

    # transform all_data to DataFrame
    rainfall_df = pd.DataFrame(all_data)

    # save the data to csv
    output_file_path = os.path.join(out_dir, 'city_info_data.csv')
    rainfall_df.to_csv(output_file_path, index=False)

    print(f"urban data is saved to {output_file_path}")
