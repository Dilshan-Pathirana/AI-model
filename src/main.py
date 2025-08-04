import os
import sys

# Ensure the src folder is in the Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.predict_future import predict_future_xgboost_solar_with_temperature_trend
from src.models.model_trainer_temp import predict_future_temperature, train_temperature_model, print_model_metrics_temp
from src.models.model_trainer import train_xgboost_model, print_model_metrics_solar 
from src.utils.fetch_data import fetch_data
from src.utils.data_cleaner import clean_data
from src.visualization.plotting import plot_historical_and_predictions
# Compute power generation
from src.models.predict_future import compute_power_generation_from_irradiance


def main():
    lat = 7.8731
    lon = 80.7718
    
    # === 1. Temperature Model Date Range ===
    temp_train_start = "1796-01-01"  # Temperature data start date
    temp_train_end = "2013-09-01"    # Temperature data end date
    future_temp_start = "2025-01-01"  # Future temperature prediction start
    future_temp_end = "2030-12-31"    # Future temperature prediction end
    
    # === 2. Solar Irradiance Model Date Range ===
    solar_train_start = "1988-01-01"  # Solar data start date (change as per actual data)
    solar_train_end = "2022-12-31"    # Solar data end date (change as per actual data)
    future_solar_start = "2025-01-01" # Future solar irradiance prediction start
    future_solar_end = "2026-01-01"   # Future solar irradiance prediction end

    # === 1. Train temperature model ===
    print("🌡️ Training temperature model...")
    mse_temp, mae_temp, rmse_temp, r2_temp = train_temperature_model( country="Sri Lanka", start_date=temp_train_start, end_date=temp_train_end
)

    # === 2. Predict future temperature ===
    print("🌍 Starting future temperature prediction for Sri Lanka...")
    predict_future_temperature(start_date=future_temp_start, end_date=future_temp_end)
    
    # === 3. Train solar irradiance model ===
    print("🔆 Training solar irradiance model...")
    mse, mae, rmse, r2 = train_xgboost_model(lat=lat, lon=lon, start_date=solar_train_start, end_date=solar_train_end)

    # === 4. Predict future solar irradiance ===
    print("🔮 Predicting future solar irradiance...")
    future_df = predict_future_xgboost_solar_with_temperature_trend(lat=lat, lon=lon, start_date=future_solar_start, end_date=future_solar_end)

    # Now that we have the predicted solar irradiance, compute power generation
    future_df = compute_power_generation_from_irradiance(future_df, irradiance_column="predicted_irradiance_trended")
    future_df.to_csv("outputs/predicted_future_solar_with_power.csv", index=False)

    # === 5. Load historical irradiance for plotting ===
    print("📊 Loading historical data for plotting...")
    historical_df = fetch_data(lat=lat, lon=lon, start_date=solar_train_start, end_date=solar_train_end)
    historical_df = clean_data(historical_df)

    # === 6. Plot combined output ===
    plot_historical_and_predictions(
        historical_df=historical_df,
        prediction_csv_path="outputs/predicted_future_solar_with_power.csv"
    )

    # === 7. Print model metrics ===
    print("📈 Printing model metrics:")
    # Now call the function to print the metrics for both models

    print_model_metrics_solar("Solar Irradiance Model", mse, mae, rmse, r2)
    print_model_metrics_temp("Temperature Model", mse_temp, mae_temp, rmse_temp, r2_temp)


if __name__ == "__main__":
    main()
    print("✅ All tasks completed successfully!")
    print("🔚 End of script.")
    print("📈 Check the outputs folder for generated plots and predictions.")
