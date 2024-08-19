import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from bs4 import BeautifulSoup
import time

# url request
def build_url(year, month, day, hour, wind, surface, level, overlay, projection, lon, lat):
    base_url = "https://earth.nullschool.net"
    url = f"{base_url}/#{year}/{int(month):02d}/{int(day):02d}/{int(hour):02d}00Z/{wind}/{surface}/{level}/overlay={overlay}/orthographic={projection[0]},{projection[1]},{projection[2]}/loc={lon},{lat}"
    return url

# get rain data
def get_rainfall_data(lon, lat, date):
    year, month, day = date.split('-')
    hour = 10
    wind = "wind"
    surface = "surface"
    level = "level"
    overlay = "total_precipitable_water"
    projection = (2.43, 51.50, 10062)

    url = build_url(year, month, day, hour, wind, surface, level, overlay, projection, lon, lat)

    # set edge engine options
    edge_options = Options()
    edge_options.add_argument('--headless')
    edge_options.add_argument('--disable-gpu')
    edge_options.add_argument('--no-sandbox')
    edge_options.add_argument('--log-level=3')

    # initialize edge driver
    service = EdgeService(EdgeChromiumDriverManager().install())
    driver = webdriver.Edge(service=service, options=edge_options)
    driver.get(url)

    # wait for the page to load
    time.sleep(10)

    # get the page html
    page_html = driver.page_source

    # parse the html
    soup = BeautifulSoup(page_html, 'html.parser')

    # find the divs containing the rainfall data
    rainfall_divs = soup.find_all('div', attrs={'aria-label': lambda x: x and 'kg/m²' in x})

    # extract the rainfall data
    rainfall_data = [div.get_text() for div in rainfall_divs]

    # close the driver
    driver.quit()

    if rainfall_data:
        return rainfall_data[0]
    else:
        return None

if __name__ == '__main__':
    # read city info data
    input_file = r'data\output\city_info_data.csv'
    out_dir=r'data\output\city'
    df = pd.read_csv(input_file)

    results = []

    for index, row in df.iterrows():
        city = row['city']
        date = row['date']
        lon = row['lon']
        lat = row['lat']
        
        output_file = os.path.join(out_dir, city, f'rainfall_data.csv')

        # check if the data already exists
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            if date in existing_df['date'].values:
                print(f"Data for {city} on {date} already exists. Skipping...")
                continue
        else:
            existing_df = pd.DataFrame()

        # get the rainfall data
        rainfall = get_rainfall_data(lon, lat, date)
        result = {'city': city, 'date': date, 'lon': lon, 'lat': lat, 'rainfall': rainfall}
        
        # save the data
        updated_df = existing_df.append(result, ignore_index=True)
        updated_df.to_csv(output_file, index=False)
        
        print(f"!!!!!!!!!!!!!!!!!!{city} {date} 降雨量: {rainfall}!!!!!!!!!!!!!!!!!!!")