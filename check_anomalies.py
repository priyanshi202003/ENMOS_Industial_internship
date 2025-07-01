import pandas as pd

# Load the data
df = pd.read_csv('data/processed/combined_data.csv')

print("Data shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nAnomaly counts:")
print("temp:", df['is_anomaly_temp'].sum())
print("current:", df['is_anomaly_current'].sum())
print("humidity:", df['is_anomaly_humidity'].sum())
print("vibration:", df['is_anomaly_vibration'].sum())
print("pressure:", df['is_anomaly_pressure'].sum())
print("viscosity:", df['is_anomaly_viscosity'].sum())
print("power:", df['is_anomaly_power'].sum())

print("\nMaintenance needed:", df['maintenance_needed'].sum())

print("\nSample of anomalies:")
anomaly_sample = df[df['is_anomaly_temp'] == 1].head()
print(anomaly_sample[['timestamp', 'temperature', 'is_anomaly_temp', 'maintenance_needed']]) 