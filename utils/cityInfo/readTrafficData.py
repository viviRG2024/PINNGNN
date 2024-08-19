import os
import pandas as pd

def get_samples_from_csv(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return len(df)
    else:
        print(f"File not found: {file_path}")
        return 0

def get_time_span_from_rainfall(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')
        
        time_spans = []
        start_date = df['date'].iloc[0]
        end_date = df['date'].iloc[0]

        for current_date in df['date'][1:]:
            if (current_date - end_date).days == 1:
                end_date = current_date
            else:
                time_spans.append(f"{start_date.strftime('%Y/%m/%d')}-{end_date.strftime('%Y/%m/%d')}")
                start_date = current_date
                end_date = current_date
        
        time_spans.append(f"{start_date.strftime('%Y/%m/%d')}-{end_date.strftime('%Y/%m/%d')}")

        # calculate those city that have rainfall > 15 
        high_rainfall_days = df[df['rainfall'] > 15].shape[0]
        return time_spans, len(df), df['rainfall'].max(), df['rainfall'].min(), high_rainfall_days
    else:
        print(f"File not found: {file_path}")
        return [], 0, None, None, 0

def process_city_folder(city_folder):
    sensor_file = os.path.join(city_folder, f"{os.path.basename(city_folder)}_sensor.csv")
    detectors_file = os.path.join(city_folder, "detectors_public.csv")
    rainfall_file = os.path.join(city_folder, "rainfall_data.csv")
    
    sensor_samples = get_samples_from_csv(sensor_file)
    detectors_samples = get_samples_from_csv(detectors_file)
    rainfall_time_spans, rainfall_samples, rainfall_max, rainfall_min, high_rainfall_days = get_time_span_from_rainfall(rainfall_file)
    
    result = {
        'city': os.path.basename(city_folder),
        'sensor_samples': sensor_samples,
        'detectors_samples': detectors_samples,
        'rainfall_samples': rainfall_samples,
        'rainfall_max': rainfall_max,
        'rainfall_min': rainfall_min,
        'high_rainfall_days': high_rainfall_days,
        'rainfall_time_spans': ', '.join(rainfall_time_spans)
    }
    
    print(f"City: {result['city']}")
    print(f"Sensor samples: {result['sensor_samples']}")
    print(f"Detectors samples: {result['detectors_samples']}")
    print(f"Rainfall samples: {result['rainfall_samples']}")
    print(f"Rainfall max: {result['rainfall_max']}")
    print(f"Rainfall min: {result['rainfall_min']}")
    print(f"High rainfall days: {result['high_rainfall_days']}")
    print(f"Rainfall time spans: {result['rainfall_time_spans']}")
    print("\n")
    
    return result

def process_all_city_folders(root_folder, output_csv):
    results = []
    for city_folder in os.listdir(root_folder):
        city_folder_path = os.path.join(root_folder, city_folder)
        if os.path.isdir(city_folder_path):
            result = process_city_folder(city_folder_path)
            if result: 
                results.append(result)
    
    # save results to output_csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    root_folder_path = r'data\output\city'  
    output_csv_path = r'data\output\city_data_summary.csv' 
    process_all_city_folders(root_folder_path, output_csv_path)

    print(f"Results have been saved to {output_csv_path}")
