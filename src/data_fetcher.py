# src/data_fetcher.py
import requests
import pandas as pd
import os
from datetime import datetime

def fetch_data_from_csv(filename):
    """
    Fetch historical climate data from a CSV file.
    """
    file_path = os.path.join('data', 'raw', filename)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        print(f"Data loaded from {file_path}")
        return data
    else:
        raise FileNotFoundError(f"File {filename} not found.")

def fetch_data_from_api(lat, lon, start_date, end_date, parameters=["ALLSKY_SFC_SW_DWN", "T2M"]):
    """
    Fetch meteorological data from NASA POWER API.
    """
    # Convert date strings from YYYY-MM-DD to YYYYMMDD format
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
        lat = float(lat)
        lon = float(lon)
    except ValueError as e:
        raise ValueError(f"Error in parsing values: {e}")

    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters={','.join(parameters)}&community=RE&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON"
    
    print(f"Fetching data from: {url}")
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"NASA API call failed: {response.status_code}")
    
    data = response.json()
    parameter_data = data['properties']['parameter']
    
    # Convert the data to a DataFrame
    df = pd.DataFrame()
    for param, values in parameter_data.items():
        temp_df = pd.DataFrame.from_dict(values, orient='index', columns=[param])
        if df.empty:
            df = temp_df
        else:
            df = df.join(temp_df)
    
    df.index = pd.to_datetime(df.index)
    
    return df


def fetch_data(lat=None, lon=None, start_date=None, end_date=None, filename=None):
    """
    Fetch data either from CSV or from API.
    """
    if filename:
        return fetch_data_from_csv(filename)
    else:
        return fetch_data_from_api(lat, lon, start_date, end_date)
