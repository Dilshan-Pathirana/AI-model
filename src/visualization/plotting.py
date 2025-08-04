import matplotlib.pyplot as plt

def plot_historical_and_predictions(historical_df=None, prediction_csv_path=None):
    import pandas as pd
    import matplotlib.pyplot as plt

    if prediction_csv_path:
        future_df = pd.read_csv(prediction_csv_path, parse_dates=["date"])
    else:
        raise ValueError("prediction_csv_path must be provided")

    if historical_df is None:
        raise ValueError("historical_df must be provided")

    # 1️⃣ Plot 1 - Historical Irradiance Only
    plt.figure(figsize=(14, 6))
    plt.plot(historical_df["date"], historical_df["ALLSKY_SFC_SW_DWN"], label="Historical Irradiance", color="blue", alpha=0.7)
    plt.title("Historical Solar Irradiance (1988–2020)")
    plt.xlabel("Date")
    plt.ylabel("Irradiance (kWh/m²/day)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/plot_historical_only.png")
    plt.show()

    # 2️⃣ Plot 2 - Future Predictions Only (Base and Trended)
    plt.figure(figsize=(14, 6))
    plt.plot(future_df["date"], future_df["daily_energy_kwh"], label="Predicted Daily Energy Output (kWh)", color="green", linestyle=":")
    plt.plot(future_df["date"], future_df["predicted_solar_irradiance"], label="Predicted (Base)", color="gray", linestyle="--")
    plt.plot(future_df["date"], future_df["predicted_irradiance_trended"], label="Predicted (Trended T2M)", color="orange")

    # Confidence interval: ±10%
    upper = future_df["predicted_irradiance_trended"] * 1.1
    lower = future_df["predicted_irradiance_trended"] * 0.9
    plt.fill_between(future_df["date"], lower, upper, color="orange", alpha=0.2, label="Confidence Interval (±10%)")

    plt.title("Future Solar Irradiance Predictions (2025–2026)")
    plt.xlabel("Date")
    plt.ylabel("Irradiance (kWh/m²/day)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/plot_future_only.png")
    plt.show()

    # 3️⃣ Plot 3 - Predicted Power Generation Only
    plt.figure(figsize=(14, 6))
    plt.plot(future_df["date"], future_df["daily_energy_kwh"], label="Predicted Daily Energy Output (kWh)", color="darkgreen")
    plt.title("Predicted Solar Power Generation (2025–2026)")
    plt.xlabel("Date")
    plt.ylabel("Energy Output (kWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/plot_power_generation_only.png")
    plt.show()
