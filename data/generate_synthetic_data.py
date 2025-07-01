import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_all_data_with_single_anomaly(timestamps, anomaly_prob=0.0005):
    """
    Generate synthetic data for all parameters, ensuring only one parameter can have an anomaly at a time per timestamp.
    """
    n = len(timestamps)
    # Decide if there is an anomaly at each timestamp
    anomaly_flags = np.random.choice([0, 1], size=n, p=[1-anomaly_prob, anomaly_prob])
    # For each anomaly, randomly assign it to a parameter
    anomaly_params = np.random.choice([
        'temperature', 'current', 'humidity', 'vibration', 'pressure', 'viscosity', 'power'
    ], size=n)

    # Generate base data for each parameter
    hours = np.array([t.hour + t.minute/60 for t in timestamps])
    days_of_week = np.array([t.weekday() for t in timestamps])

    # Temperature
    base_temp = 25 + 5 * np.sin(2 * np.pi * hours / 24)
    weekly_temp = np.where(days_of_week >= 5, -2, 0)
    temp_noise = np.random.normal(0, 0.5, n)
    temp_anomaly = np.random.normal(0, 3, n) * (anomaly_flags * (anomaly_params == 'temperature'))
    temperature = base_temp + weekly_temp + temp_noise + temp_anomaly
    is_anomaly_temp = (anomaly_flags & (anomaly_params == 'temperature'))

    # Current
    base_current = 5 + 2 * np.sin(2 * np.pi * hours / 24)
    weekly_current = np.where(days_of_week >= 5, -1, 0)
    current_noise = np.random.normal(0, 0.2, n)
    current_anomaly = np.random.normal(0, 2, n) * (anomaly_flags * (anomaly_params == 'current'))
    current = np.maximum(base_current + weekly_current + current_noise + current_anomaly, 0)
    is_anomaly_current = (anomaly_flags & (anomaly_params == 'current'))

    # Humidity
    base_humidity = 60 + 10 * np.sin(2 * np.pi * hours / 24)
    weekly_humidity = np.where(days_of_week >= 5, 5, 0)
    humidity_noise = np.random.normal(0, 2, n)
    humidity_anomaly = np.random.normal(0, 15, n) * (anomaly_flags * (anomaly_params == 'humidity'))
    humidity = np.clip(base_humidity + weekly_humidity + humidity_noise + humidity_anomaly, 0, 100)
    is_anomaly_humidity = (anomaly_flags & (anomaly_params == 'humidity'))

    # Vibration
    base_vibration = 0.5 + 0.2 * np.sin(2 * np.pi * hours / 24)
    vibration_noise = np.random.normal(0, 0.1, n)
    vibration_anomaly = np.random.exponential(1.0, n) * (anomaly_flags * (anomaly_params == 'vibration'))
    vibration = np.maximum(base_vibration + vibration_noise + vibration_anomaly, 0)
    is_anomaly_vibration = (anomaly_flags & (anomaly_params == 'vibration'))

    # Pressure
    base_pressure = 100 + 10 * np.sin(2 * np.pi * hours / 24)
    pressure_noise = np.random.normal(0, 2, n)
    pressure_anomaly = np.random.normal(0, 20, n) * (anomaly_flags * (anomaly_params == 'pressure'))
    pressure = np.clip(base_pressure + pressure_noise + pressure_anomaly, 50, 150)
    is_anomaly_pressure = (anomaly_flags & (anomaly_params == 'pressure'))

    # Viscosity
    base_viscosity = 50 + 5 * np.sin(2 * np.pi * hours / 24)
    viscosity_noise = np.random.normal(0, 1, n)
    viscosity_anomaly = np.random.normal(0, 10, n) * (anomaly_flags * (anomaly_params == 'viscosity'))
    viscosity = np.clip(base_viscosity + viscosity_noise + viscosity_anomaly, 20, 100)
    is_anomaly_viscosity = (anomaly_flags & (anomaly_params == 'viscosity'))

    # Power
    voltage = 220
    base_power = current * voltage
    power_noise = np.random.normal(0, 100, n)
    power_anomaly = np.random.normal(0, 500, n) * (anomaly_flags * (anomaly_params == 'power'))
    power = np.maximum(base_power + power_noise + power_anomaly, 0)
    is_anomaly_power = (anomaly_flags & (anomaly_params == 'power'))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'is_anomaly_temp': is_anomaly_temp,
        'current': current,
        'is_anomaly_current': is_anomaly_current,
        'humidity': humidity,
        'is_anomaly_humidity': is_anomaly_humidity,
        'vibration': vibration,
        'is_anomaly_vibration': is_anomaly_vibration,
        'pressure': pressure,
        'is_anomaly_pressure': is_anomaly_pressure,
        'viscosity': viscosity,
        'is_anomaly_viscosity': is_anomaly_viscosity,
        'power': power,
        'is_anomaly_power': is_anomaly_power
    })
    return df

def generate_maintenance_data(temperature_df, current_df, humidity_df, vibration_df, 
                            pressure_df, viscosity_df, power_df):
    """
    Generate synthetic maintenance data based on all parameters
    """
    # Combine all data
    combined_data = pd.merge(
        pd.merge(
            pd.merge(
                pd.merge(
                    pd.merge(
                        pd.merge(
                            pd.merge(temperature_df, current_df, on='timestamp'),
                            humidity_df, on='timestamp'
                        ),
                        vibration_df, on='timestamp'
                    ),
                    pressure_df, on='timestamp'
                ),
                viscosity_df, on='timestamp'
            ),
            power_df, on='timestamp'
        ),
        maintenance_df, on='timestamp'
    )
    
    # Generate maintenance flags based on conditions - more realistic thresholds
    maintenance_needed = (
        (combined_data['temperature'] > 35) |      # Higher temperature threshold
        (combined_data['current'] > 8) |           # Higher current threshold
        (combined_data['humidity'] > 90) |         # Higher humidity threshold
        (combined_data['vibration'] > 1.5) |       # Higher vibration threshold
        (combined_data['pressure'] > 140) |        # Higher pressure threshold
        (combined_data['viscosity'] > 85) |        # Higher viscosity threshold
        (combined_data['power'] > 2500) |          # Higher power threshold
        (combined_data['is_anomaly_temp'] | 
         combined_data['is_anomaly_current'] | 
         combined_data['is_anomaly_humidity'] |
         combined_data['is_anomaly_vibration'] |
         combined_data['is_anomaly_pressure'] |
         combined_data['is_anomaly_viscosity'] |
         combined_data['is_anomaly_power'])        # Any anomaly
    )
    
    # Add maintenance probability - more realistic probabilities
    maintenance_prob = np.where(maintenance_needed, 0.95, 0.001)  # 95% if needed, 0.1% if not
    maintenance_flag = np.random.binomial(1, maintenance_prob)
    
    return pd.DataFrame({
        'timestamp': combined_data['timestamp'],
        'maintenance_needed': maintenance_flag,
        'maintenance_probability': maintenance_prob
    })

def save_dataframe(df, filepath):
    """Save DataFrame to CSV with error checking"""
    try:
        df.to_csv(filepath, index=False)
        print(f"Successfully saved {len(df)} rows to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving {filepath}: {str(e)}")
        return False

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate timestamps ONCE
    days = 30
    start_time = datetime.now() - timedelta(days=days)
    timestamps = [start_time + timedelta(minutes=i) for i in range(days * 24 * 60)]
    
    # Generate data
    print("Generating all data with single anomaly...")
    all_data_df = generate_all_data_with_single_anomaly(timestamps)
    if not save_dataframe(all_data_df, 'data/raw/all_data.csv'):
        return
    
    # Create combined dataset for ML training
    print("Creating combined dataset...")
    combined_df = all_data_df.copy()
    if not save_dataframe(combined_df, 'data/processed/combined_data.csv'):
        return
    
    print("\nData generation complete!")
    print(f"Generated {len(all_data_df)} data points")
    print(f"Number of anomalies: {sum(combined_df[col].sum() for col in combined_df.columns if 'is_anomaly' in col)}")
    
    # Verify the data was saved correctly
    print("\nVerifying saved data...")
    for filepath in ['data/raw/all_data.csv', 'data/processed/combined_data.csv']:
        try:
            df = pd.read_csv(filepath)
            print(f"{filepath}: {len(df)} rows")
        except Exception as e:
            print(f"Error reading {filepath}: {str(e)}")

if __name__ == "__main__":
    main() 