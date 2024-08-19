import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from concave_hull import concave_hull_indexes

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def merge_geojson_files(city_folder):
    # initialize an empty list to store GeoDataFrames
    gdfs = []
    
    # iterate over all files in the city folder
    for file in os.listdir(city_folder):
        if file.endswith('.gpkg'): 
            file_path = os.path.join(city_folder, file)
            # read the GeoDataFrame from the file
            gdf = gpd.read_file(file_path, layer="edges")
            gdfs.append(gdf)
    
    # combine all GeoDataFrames into a single GeoDataFrame
    if gdfs:
        combined_gdf = pd.concat(gdfs, ignore_index=True)
        return combined_gdf
    else:
        return None

def process_city(city_folder):
    # read sensors location.csv
    links_path = os.path.join(city_folder, 'detectors_public.csv')
    links_df = pd.read_csv(links_path)
    points_gdf = gpd.GeoDataFrame(
        links_df, geometry=gpd.points_from_xy(links_df.long, links_df.lat), crs="EPSG:4326"
    )
    
    # read road network
    road_gdf = merge_geojson_files(os.path.join(city_folder))
    
    # Convert the road GeoDataFrame to the same CRS as the points GeoDataFrame
    road_gdf = road_gdf.to_crs(points_gdf.crs)

    # Extract points from the GeoDataFrame
    points = np.array([[point.x, point.y] for point in points_gdf.geometry])
    
    # Calculate concave hull indexes
    idxes = concave_hull_indexes(
        points,
        length_threshold=0,
    )
    idxes = np.append(idxes, idxes[0])
    
    # Create a concave hull polygon
    concave_hull_points = points[idxes]
    concave_hull = Polygon(concave_hull_points)

    # Select roads that intersect or are contained by the concave hull
    selected_roads = road_gdf[road_gdf.intersects(concave_hull) | road_gdf.within(concave_hull)]
    gdf_projected = selected_roads.to_crs(epsg=32650)
    
    # visualize
    fig, ax = plt.subplots(figsize=(10, 10))
    points_gdf.plot(ax=ax, color='red', markersize=5)
    gpd.GeoSeries([concave_hull]).plot(ax=ax, facecolor='none', edgecolor='blue')
    gdf_projected.to_crs(points_gdf.crs).plot(ax=ax, color='green', linewidth=0.5)
    plt.title(f"Concave Hull and Selected Roads for {os.path.basename(city_folder)}")
    plt.savefig(os.path.join(city_folder, 'concave_hull_visualization.png'))
    plt.close()

    return gdf_projected

if __name__ == "__main__":
    config = load_config(r'config.json')
    cities = config['cities']

    base_folder = 'data\input\cities'
    for city in cities:
        city_path = os.path.join(base_folder, city)
        if os.path.isdir(city_path):
            print(f"Processing {city_path}...")
            final_network = process_city(city_path)
            
            # save the selected network
            output_path = os.path.join(city_path, f"selected_network.shp")
            final_network.to_file(output_path)
            print(f"Processed network saved to {output_path}")