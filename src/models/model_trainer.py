import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from src.utils.fetch_data import fetch_data
import joblib

def print_model_metrics(model_name, mse, mae, rmse, r2):
    """
    Prints the model evaluation metrics in a readable format with labels.
    """
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print("-" * 40)

def train_xgboost_model(lat, lon, start_date, end_date):
    """
    Train an XGBoost model using climate data fetched from the NASA POWER API.
    Returns evaluation metrics: mse, mae, rmse, r2
    """
    # Fetch the data from the API
    print("🌍 Fetching data from the API...")
    df = fetch_data(lat=lat, lon=lon, start_date=start_date, end_date=end_date)

    # Preprocess the data
    print("🔄 Preprocessing data...")
    df = preprocess_data(df)
    
    # Prepare features and target variable
    X = df.drop(columns=["ALLSKY_SFC_SW_DWN"])  # All sky surface shortwave radiation is the target
    y = df["ALLSKY_SFC_SW_DWN"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the XGBoost model
    print("🚀 Training XGBoost model...")
    model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, objective='reg:squarederror', n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    
    # Calculate common regression metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Print the model metrics
    print_model_metrics("XGBoost Model", mse, mae, rmse, r2)

    # Save the trained model
    model_path = os.path.join('models', 'xgboost_model.json')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"✅ Model saved at: {model_path}")

    # Save the scaler for future use
    scaler_path = os.path.join('models', 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved at: {scaler_path}")

    # ✅ Return metrics
    return mse, mae, rmse, r2

def preprocess_data(df):
    """
    Preprocess the data: fill missing values and generate necessary features.
    """
    # Fill missing values with the median of each column
    df.fillna(df.median(), inplace=True)

    # Create additional features if needed (e.g., time-based features)
    # Example: Adding cyclical features for 'Date' (if date is available)
    if 'date' in df.columns:
        df['dayofyear'] = df['date'].dt.dayofyear
        df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        df.drop(columns=['date', 'dayofyear'], inplace=True)

    # Time-based feature: Previous day's irradiance (lag feature)
    df['prev_irradiance'] = df['ALLSKY_SFC_SW_DWN'].shift(1)
    df.dropna(inplace=True)  # Drop rows with NaN values after shifting
    
    return df

def print_model_metrics_solar(model_name, mse, mae, rmse, r2):
    """
    Prints the model evaluation metrics in a readable format with labels.
    """
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print("-" * 40)