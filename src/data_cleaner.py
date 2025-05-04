import numpy as np
import pandas as pd

def clean_data(df):
    """
    Clean the fetched data by handling missing values and selecting relevant columns.
    """
    # Drop rows with missing values
    df = df.dropna()
    
    # Ensure the DataFrame has a 'date' column, if not, create one from the index (if it's a time series)
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df.index)

    # Select only relevant columns (temperature and solar irradiance)
    df = df[['T2M', 'ALLSKY_SFC_SW_DWN', 'date']]  # Ensure 'date' is included
    
    # Resample by day if needed
    df = df.resample('D').mean()
    
    # Call add_cyclical_features to add cyclical components to time-related data
    df = add_cyclical_features(df)
    
    return df

def add_cyclical_features(data):
    """
    Adds cyclical features for 'day_of_year' and 'month' to the DataFrame.
    The cyclical features represent the repeating nature of time (e.g., months and days of the year).
    """
    # Ensure 'date' is datetime
    if not isinstance(data['date'].dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        data['date'] = pd.to_datetime(data['date'])

    # Add 'day_of_year' and 'month' columns
    data['day_of_year'] = data['date'].dt.dayofyear
    data['month'] = data['date'].dt.month

    # Cyclical transformations for 'day_of_year'
    data['sin_day_of_year'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * data['day_of_year'] / 365)

    # Cyclical transformations for 'month'
    data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12)
    data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12)

    return data
