import numpy as np
import pandas as pd

def add_cyclical_features(df):
    """
    Adds cyclical features for the 'date' column (e.g., day of the year, month).
    """
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    
    # Convert day_of_year and month into cyclic features
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df
