import pandas as pd
import matplotlib.pyplot as plt
import os

def inspect_file(filepath, n=5):
    print(f"\nInspecting {filepath}:")
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    print(df.head(n))
    print("...")
    print(df.tail(n))
    print("\nSummary stats:")
    print(df.describe(include='all'))
    return df

def plot_time_series(df, value_col, anomaly_col=None, maintenance_col=None, title=None):
    plt.figure(figsize=(15, 5))
    plt.plot(df['timestamp'], df[value_col], label=value_col, color='blue')
    if anomaly_col is not None:
        anomalies = df[df[anomaly_col] == 1]
        plt.scatter(anomalies['timestamp'], anomalies[value_col], color='red', label='Anomaly', s=10)
    if maintenance_col is not None:
        maintenance = df[df[maintenance_col] == 1]
        plt.scatter(maintenance['timestamp'], maintenance[value_col], color='orange', label='Maintenance', s=10, marker='x')
    plt.title(title or value_col)
    plt.xlabel('Timestamp')
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Inspect files
    temp_df = inspect_file('data/raw/temperature_data.csv')
    current_df = inspect_file('data/raw/current_data.csv')
    maintenance_df = inspect_file('data/raw/maintenance_data.csv')
    combined_df = inspect_file('data/processed/combined_data.csv')

    # Visualize temperature
    plot_time_series(temp_df, 'temperature', anomaly_col='is_anomaly_temp', title='Temperature with Anomalies')
    # Visualize current
    plot_time_series(current_df, 'current', anomaly_col='is_anomaly_current', title='Current with Anomalies')
    # Visualize maintenance events (on temperature)
    plot_time_series(combined_df, 'temperature', anomaly_col='is_anomaly_temp', maintenance_col='maintenance_needed', title='Temperature, Anomalies, and Maintenance')
    # Visualize maintenance events (on current)
    plot_time_series(combined_df, 'current', anomaly_col='is_anomaly_current', maintenance_col='maintenance_needed', title='Current, Anomalies, and Maintenance')

 