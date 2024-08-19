import os
import pandas as pd

if __name__ == '__main__':
    in_dir = r'data\output\city' 

    all_data = []

    for city_folder in os.listdir(in_dir):
        city_folder_path = os.path.join(in_dir, city_folder)
        file_path=os.path.join(city_folder_path, f'rainfall_data.csv')
        if not os.path.exists(file_path):
            print(f"{file_path} does not exist")
            continue
        else:
            df=pd.read_csv(file_path)
            row=df.shape[0]
            if row>=10:
                print(f"city:{city_folder},row:{row}")
                all_data.append(city_folder)
    
    print(len(all_data))
    print(all_data)
        