from data.data_loader import fetch_data_from_csv, fetch_data_from_api

def fetch_data(lat=None, lon=None, start_date=None, end_date=None, filename=None):
    """
    Fetch data either from CSV or from NASA POWER API.
    """
    if filename:
        print(f"Fetching data from CSV: {filename}")
        return fetch_data_from_csv(filename)
    elif lat and lon and start_date and end_date:
        print(f"Fetching data from NASA API for lat={lat}, lon={lon}, from {start_date} to {end_date}")
        return fetch_data_from_api(lat, lon, start_date, end_date)
    else:
        raise ValueError("❌ Either 'filename' or (lat, lon, start_date, end_date) must be provided.")
