import pandas as pd
import os
import shutil

if __name__ == '__main__':
    in_dir=r'D:\OneDrive\Documents\002_UCL\005_Phd\003_code\002_data\002_traffic_count\eth_dataset'
    out_dir=r'data\output\city'

    links_df = pd.read_csv(os.path.join(in_dir,'links.csv'))
    detectors_df = pd.read_csv(os.path.join(in_dir,'detectors_public.csv'))

    # get all city codes
    city_codes = set(links_df['citycode']).union(set(detectors_df['citycode']))

    # create a folder for each city code
    for city_code in city_codes:
        # create folder
        folder_name = os.path.join(out_dir, f'{city_code}')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # get data for each city code
        links_city_df = links_df[links_df['citycode'] == city_code]
        detectors_city_df = detectors_df[detectors_df['citycode'] == city_code]

        # save data to CSV files
        links_city_df.to_csv(os.path.join(folder_name, 'links.csv'), index=False)
        detectors_city_df.to_csv(os.path.join(folder_name, 'detectors_public.csv'), index=False)

        print(f"data saved to {folder_name}")

    # move all CSV files to a new folder
    other_csv_folder = r'data\output\bk'

    import dask.dataframe as dd
    import os

    in_path=r'D:\OneDrive\Documents\002_UCL\005_Phd\003_code\002_data\002_traffic_count\eth_dataset\utd19_u.csv'
    out_dir=r'data\output\city'

    # read the CSV file
    df = dd.read_csv(in_path, assume_missing=True)

    # get all unique cities
    cities = df['city'].unique().compute()

    # create a folder for each city
    for city in cities:
        city_df = df[df['city'] == city]
        out_path=os.path.join(out_dir, city, f'{city}_sensor.csv')
        if not os.path.exists(out_path):
            city_df.to_csv(out_path, single_file=True)
        print(f'{city}_data.csv has been saved!')

