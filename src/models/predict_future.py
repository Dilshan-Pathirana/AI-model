import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

def fetch_future_temperature(start_date="2025-01-01", end_date="2030-12-31"):
    """
    Load future temperature data from the predicted file, filtered by date range.
    Assumes temperature is a general value (not per lat/lon).
    """
    path = "outputs/predicted_future_temperature.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Future temperature data not found at {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
    df = df.rename(columns={"predicted_temperature": "T2M"})  # Standardize name
    return df

def generate_features_for_future_data(df):
    """
    Generate necessary features like prev_irradiance, sin/cos dayofyear for future data.
    Assumes that `df` has a 'date' column and 'T2M' column (temperature).
    """
    # Generate cyclical features (day of year)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["sin_dayofyear"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["cos_dayofyear"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)

    # Generate the 'prev_irradiance' feature (previous day's irradiance)
    df['prev_irradiance'] = df['T2M'].shift(1)  # Assuming T2M correlates with irradiance

    # Drop rows with NaN values due to shifting (first row will have NaN for prev_irradiance)
    df = df.dropna(subset=['prev_irradiance'])
    
    return df


def apply_temperature_trend(df):
    """
    Add linear trend to T2M and compute updated prediction.
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["ordinal"] = df["date"].map(datetime.toordinal)

    # Trend for T2M
    t_model = LinearRegression()
    t_model.fit(df[["ordinal"]], df["T2M"])
    df["T2M_trended"] = t_model.predict(df[["ordinal"]])

    return df

def predict_future_xgboost_solar_with_temperature_trend(lat, lon, start_date="2025-01-01", end_date="2030-12-31"):
    print("🔆 Starting trended solar irradiance prediction...")

    # Load model and scaler
    model_path = os.path.join("models", "xgboost_model.json")
    scaler_path = os.path.join("models", "scaler.pkl")
    model = XGBRegressor()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Load future temperature data
    df_temp = fetch_future_temperature(start_date, end_date)
    df_temp = generate_features_for_future_data(df_temp)
    df_temp["lat"] = lat
    df_temp["lon"] = lon

    # Add dummy prev_irradiance (use rolling mean of T2M or another proxy if needed)
    df_temp["prev_irradiance"] = df_temp["T2M"].shift(1)
    df_temp["prev_irradiance"].fillna(method="bfill", inplace=True)

    # Select features in the correct order
    features = df_temp[["T2M", "sin_dayofyear", "cos_dayofyear", "prev_irradiance"]]
    features_scaled = scaler.transform(features)

    # Predict with raw T2M
    df_temp["predicted_solar_irradiance"] = model.predict(features_scaled)

    # Apply trend to T2M
    df_temp = apply_temperature_trend(df_temp)

    # Replace T2M with T2M_trended and reassemble features
    trended_features = df_temp[["T2M_trended", "sin_dayofyear", "cos_dayofyear", "prev_irradiance"]]
    trended_features.columns = ["T2M", "sin_dayofyear", "cos_dayofyear", "prev_irradiance"]  # Ensure names match
    trended_scaled = scaler.transform(trended_features)

    # Predict with trended temperature
    df_temp["predicted_irradiance_trended"] = model.predict(trended_scaled)

    # Save predictions
    df_temp.to_csv("outputs/predicted_future_solar_with_temp_trend.csv", index=False)
    print("✅ Saved with temperature trend comparison.")

    return df_temp

def compute_power_generation_from_irradiance(df, irradiance_column="predicted_irradiance_trended", panel_area=1.6, efficiency=0.20, performance_ratio=0.75):
    """
    Estimate daily energy output from solar panel based on predicted irradiance.
    """
    df["daily_energy_kwh"] = df[irradiance_column] * panel_area * efficiency * performance_ratio
    print("Columns in DataFrame after power calculation:", df.columns)  # Debugging line
    return df

