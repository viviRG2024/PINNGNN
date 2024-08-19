import os
import osmnx as ox


if __name__ == '__main__':
    city_folder_path = r'data\input\cities'
    cities = [name for name in os.listdir(city_folder_path) if os.path.isdir(os.path.join(city_folder_path, name))]

    # download road data for each city
    for city in cities:
        # create folder for each city
        folder_name = os.path.join(city_folder_path, city)
        file_path = os.path.join(folder_name, 'roads.gpkg')
        if not os.path.exists(file_path):
            # download road data
            try:
                G = ox.graph_from_place(city, network_type='all')
                
                # save the road data as GeoPackage file
                
                ox.save_graph_geopackage(G, filepath=file_path, encoding='utf-8')
                
                print(f"{city} saved to {file_path}")
            except Exception as e:
                print(f"can not get {city}'s road data: {e}")
        else:
            print(f"{city} saved to' {file_path}")
