import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def print_model_metrics_temp(model_name, mse, mae, rmse, r2):
    """
    Prints the model evaluation metrics in a readable format with labels.
    """
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print("-" * 40)

def fetch_temperature_data(country="Sri Lanka", start_date="1796-01-01", end_date="2013-09-01"):
    """
    Fetch historical temperature data from a local CSV file, filtered by country and date.
    """
    file_path = r"E:\Cademics\research\AI model\data\raw\GlobalLandTemperaturesByCountry.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found at {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    # Convert 'dt' column to datetime
    if 'dt' in df.columns:
        df['date'] = pd.to_datetime(df['dt'])
    else:
        raise ValueError("❌ Column 'dt' not found in dataset")

    # Filter by country and date
    df = df[(df['Country'] == country)]
    df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

    # Drop rows with missing temperature values
    df = df.dropna(subset=['AverageTemperature'])

    # Keep only relevant columns
    df = df[['date', 'AverageTemperature']]

    return df

def preprocess_temperature_data(df):
    """
    Preprocess the temperature data: fill missing values and generate necessary features.
    """
    # Drop rows with missing temperature
    df.dropna(subset=["AverageTemperature"], inplace=True)

    # Create cyclical features from date
    df['dayofyear'] = df['date'].dt.dayofyear
    df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    # Lag feature: previous day's temperature
    df['prev_temperature'] = df['AverageTemperature'].shift(1)
    df.dropna(inplace=True)

    # Keep only necessary columns
    return df[["date", "AverageTemperature", "sin_dayofyear", "cos_dayofyear", "prev_temperature"]]

def train_temperature_model(country="Sri Lanka", start_date="1796-01-01", end_date="2013-09-01"):
    """
    Train an XGBoost model using the temperature data from a local CSV file.
    """
    print("🌍 Loading local temperature data...")
    df = fetch_temperature_data(country=country, start_date=start_date, end_date=end_date)

    print(f"Data loaded with {df.shape[0]} records.")

    if df.empty:
        raise ValueError(f"❌ No data available for the given filters: Country: {country}, Date Range: {start_date} to {end_date}")
    
    print("🔄 Preprocessing data...")
    df = preprocess_temperature_data(df)

    # Check if data is empty after preprocessing
    if df.empty:
        raise ValueError("❌ No data available after preprocessing.")

    # Define features and target
    X = df[["sin_dayofyear", "cos_dayofyear", "prev_temperature"]]
    y = df["AverageTemperature"]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("🚀 Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        objective='reg:squarederror',
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print_model_metrics_temp("Temperature Model", mse, mae, rmse, r2)




    print(f"✅ Model trained with MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Save model
    model_path = os.path.join('models', 'temperature_xgboost_model.json')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"✅ Model saved at: {model_path}")

    # Save scaler
    scaler_path = os.path.join('models', 'temperature_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved at: {scaler_path}")

    return mse, mae, rmse, r2

def predict_future_temperature(start_date="2025-01-01", end_date="2030-12-31"):
    """
    Predict future temperature using the trained model and scaler.
    """
    print("🚀 Starting future temperature prediction...")

    # Load trained model and scaler
    model_path = os.path.join("models", "temperature_xgboost_model.json")
    scaler_path = os.path.join("models", "temperature_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ Scaler not found at {scaler_path}")

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Generate future dates
    future_dates = pd.date_range(start=start_date, end=end_date)
    records = []

    # Start with the last known temperature value (using current historical data)
    last_known_temp = fetch_temperature_data(country="Sri Lanka", start_date="2000-01-01", end_date="2020-12-31").iloc[-1]["AverageTemperature"]

    # Predict future temperature recursively
    prev_temp = last_known_temp
    for d in future_dates:
        day_of_year = d.timetuple().tm_yday
        sin_dayofyear = np.sin(2 * np.pi * day_of_year / 365)
        cos_dayofyear = np.cos(2 * np.pi * day_of_year / 365)

        features = scaler.transform([[sin_dayofyear, cos_dayofyear, prev_temp]])
        prediction = model.predict(features)[0]
        records.append({"date": d, "predicted_temperature": prediction})

        # Update previous temperature for next prediction
        prev_temp = prediction

    df_future = pd.DataFrame(records)

    # Save predicted future temperatures to CSV
    output_path = "outputs/predicted_future_temperature.csv"
    os.makedirs("outputs", exist_ok=True)
    df_future.to_csv(output_path, index=False)

    print(f"✅ Prediction complete. Future temperatures saved at: {output_path}")
    return df_future
