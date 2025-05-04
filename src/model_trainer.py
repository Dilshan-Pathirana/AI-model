from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd
import numpy as np

def train_model(historical_features, historical_target):
    """
    Train a Random Forest model with the given features and target.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(historical_features, historical_target)
    
    return model

def save_model(model, filename):
    """
    Save the trained model to disk.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def add_cyclical_features(df):
    """
    Add cyclical features (sin and cos transformations) for day of year and month.
    This helps the model learn seasonal trends.
    """
    # Add a "day of year" feature
    df['day_of_year'] = df.index.dayofyear
    # Add a "month" feature
    df['month'] = df.index.month

    # Create cyclical features using sine and cosine transformations
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

    return df
