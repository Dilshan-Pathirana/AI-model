import pandas as pd
import matplotlib.pyplot as plt
import joblib
from src.data_fetcher import fetch_data
from src.model_trainer import train_model, save_model
from src.data_cleaner import add_cyclical_features, clean_data
from datetime import datetime, timedelta
import numpy as np


def predict_future(start_year=2030, end_year=2040, lat=7.8731, lon=80.7718, historical_start_date="2020-01-01", historical_end_date="2020-12-31"):
    # Fetch historical data and process it
    print("Fetching historical climate data...")
    historical_data_raw = fetch_data(lat=lat, lon=lon, start_date=historical_start_date, end_date=historical_end_date)
    historical_data = clean_data(historical_data_raw)
    
    # Add cyclical features (Day of year, month, etc.)
    historical_data = add_cyclical_features(historical_data)
    
    # Extract features and target for training
    historical_features = historical_data[['T2M', 'day_of_year', 'month', 'sin_day_of_year', 'cos_day_of_year', 'sin_month', 'cos_month']]
    historical_target = historical_data['ALLSKY_SFC_SW_DWN']
    
    # Train the model
    print("Training Random Forest model...")
    model = train_model(historical_features, historical_target)
    
    # Save the trained model
    model_filename = 'model/random_forest_solar_model.pkl'
    save_model(model, model_filename)

    # Generate future dates
    print(f"Generating future dates from {start_year} to {end_year}...")
    future_dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='D')

    # Create DataFrame for future features (use average T2M for simplicity, or extrapolate)
    future_features = pd.DataFrame({
        'T2M': np.full(len(future_dates), historical_data['T2M'].mean()),  # Set average temperature (or extrapolate)
        'day_of_year': future_dates.dayofyear,
        'month': future_dates.month,
        'sin_day_of_year': np.sin(2 * np.pi * future_dates.dayofyear / 365),
        'cos_day_of_year': np.cos(2 * np.pi * future_dates.dayofyear / 365),
        'sin_month': np.sin(2 * np.pi * future_dates.month / 12),
        'cos_month': np.cos(2 * np.pi * future_dates.month / 12)
    })

    # Predict future solar irradiance
    future_predictions = model.predict(future_features)

    # Plot Historical Data (optional)
    plt.figure(figsize=(10, 6))
    plt.plot(historical_data.index, historical_data['ALLSKY_SFC_SW_DWN'], label='Historical Solar Irradiance', color='blue')
    plt.title(f"Historical Solar Irradiance ({historical_start_date} to {historical_end_date}) at ({lat}, {lon})")
    plt.xlabel("Date")
    plt.ylabel("Solar Irradiance (W/m²)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/historical_data.png')
    plt.show()

    # Plot Future Predictions
    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, future_predictions, label='Predicted Future Irradiance', color='orange')
    plt.title(f"Predicted Solar Irradiance ({start_year}-{end_year}) at ({lat}, {lon})")
    plt.xlabel("Date")
    plt.ylabel("Solar Irradiance (W/m²)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/future_predictions.png')
    plt.show()

    # Save predictions to CSV
    future_predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Irradiance': future_predictions
    })
    future_predictions_df.to_csv(f'outputs/future_predictions_{start_year}_{end_year}.csv', index=False)
    print(f"Future predictions saved to 'outputs/future_predictions_{start_year}_{end_year}.csv'.")
