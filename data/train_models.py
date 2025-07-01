import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import json

# Add parent directory to path to import ML models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_models.anomaly_detection import AnomalyDetector
from ml_models.predictive_maintenance import PredictiveMaintenance
from utils.data_processor import prepare_time_series_data

def load_and_prepare_data():
    """Load and prepare data for training"""
    print("Loading data from CSV...")
    # Load combined data
    data = pd.read_csv('data/processed/combined_data.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    print(f"Loaded {len(data)} data points")
    
    # Prepare features for anomaly detection
    print("\nPreparing temperature features...")
    temp_data = data[['timestamp', 'temperature']].rename(columns={'temperature': 'value'})
    temp_features = prepare_time_series_data(temp_data)
    print(f"Temperature features shape: {temp_features.shape}")
    
    print("\nPreparing current features...")
    current_data = data[['timestamp', 'current']].rename(columns={'current': 'value'})
    current_features = prepare_time_series_data(current_data)
    print(f"Current features shape: {current_features.shape}")
    
    print("\nPreparing humidity features...")
    humidity_data = data[['timestamp', 'humidity']].rename(columns={'humidity': 'value'})
    humidity_features = prepare_time_series_data(humidity_data)
    print(f"Humidity features shape: {humidity_features.shape}")
    
    print("\nPreparing vibration features...")
    vibration_data = data[['timestamp', 'vibration']].rename(columns={'vibration': 'value'})
    vibration_features = prepare_time_series_data(vibration_data)
    print(f"Vibration features shape: {vibration_features.shape}")
    
    print("\nPreparing pressure features...")
    pressure_data = data[['timestamp', 'pressure']].rename(columns={'pressure': 'value'})
    pressure_features = prepare_time_series_data(pressure_data)
    print(f"Pressure features shape: {pressure_features.shape}")
    
    print("\nPreparing viscosity features...")
    viscosity_data = data[['timestamp', 'viscosity']].rename(columns={'viscosity': 'value'})
    viscosity_features = prepare_time_series_data(viscosity_data)
    print(f"Viscosity features shape: {viscosity_features.shape}")
    
    print("\nPreparing power features...")
    power_data = data[['timestamp', 'power']].rename(columns={'power': 'value'})
    power_features = prepare_time_series_data(power_data)
    print(f"Power features shape: {power_features.shape}")
    
    # Prepare features for predictive maintenance
    print("\nPreparing maintenance features...")
    maintenance_features = np.column_stack([
        temp_features,     # Temperature features
        current_features,  # Current features
        humidity_features, # Humidity features
        vibration_features,# Vibration features
        pressure_features, # Pressure features
        viscosity_features,# Viscosity features
        power_features    # Power features
    ])
    print(f"Maintenance features shape: {maintenance_features.shape}")
    
    # Prepare targets (skip first 24 points due to window)
    temp_anomalies = data['is_anomaly_temp'].values[24:]
    current_anomalies = data['is_anomaly_current'].values[24:]
    humidity_anomalies = data['is_anomaly_humidity'].values[24:]
    vibration_anomalies = data['is_anomaly_vibration'].values[24:]
    pressure_anomalies = data['is_anomaly_pressure'].values[24:]
    viscosity_anomalies = data['is_anomaly_viscosity'].values[24:]
    power_anomalies = data['is_anomaly_power'].values[24:]
    maintenance_targets = data['maintenance_needed'].values[24:]
    
    print(f"\nTarget shapes:")
    print(f"Temperature anomalies: {temp_anomalies.shape}")
    print(f"Current anomalies: {current_anomalies.shape}")
    print(f"Humidity anomalies: {humidity_anomalies.shape}")
    print(f"Vibration anomalies: {vibration_anomalies.shape}")
    print(f"Pressure anomalies: {pressure_anomalies.shape}")
    print(f"Viscosity anomalies: {viscosity_anomalies.shape}")
    print(f"Power anomalies: {power_anomalies.shape}")
    print(f"Maintenance targets: {maintenance_targets.shape}")
    
    return {
        'temp_features': temp_features,
        'current_features': current_features,
        'humidity_features': humidity_features,
        'vibration_features': vibration_features,
        'pressure_features': pressure_features,
        'viscosity_features': viscosity_features,
        'power_features': power_features,
        'maintenance_features': maintenance_features,
        'temp_anomalies': temp_anomalies,
        'current_anomalies': current_anomalies,
        'humidity_anomalies': humidity_anomalies,
        'vibration_anomalies': vibration_anomalies,
        'pressure_anomalies': pressure_anomalies,
        'viscosity_anomalies': viscosity_anomalies,
        'power_anomalies': power_anomalies,
        'maintenance_targets': maintenance_targets
    }

def train_models():
    """Train all ML models"""
    print("Loading and preparing data...")
    data = load_and_prepare_data()
    
    # Verify data shapes before training
    print("\nVerifying data shapes before training:")
    print(f"Temperature features: {data['temp_features'].shape}")
    print(f"Current features: {data['current_features'].shape}")
    print(f"Humidity features: {data['humidity_features'].shape}")
    print(f"Vibration features: {data['vibration_features'].shape}")
    print(f"Pressure features: {data['pressure_features'].shape}")
    print(f"Viscosity features: {data['viscosity_features'].shape}")
    print(f"Power features: {data['power_features'].shape}")
    print(f"Maintenance features: {data['maintenance_features'].shape}")
    
    # Train temperature anomaly detector
    print("\nTraining temperature anomaly detector...")
    temp_detector = AnomalyDetector()
    temp_detector.train_isolation_forest(data['temp_features'])
    temp_detector.save_models('models/temperature_anomaly')
    
    # Train current anomaly detector
    print("\nTraining current anomaly detector...")
    current_detector = AnomalyDetector()
    current_detector.train_isolation_forest(data['current_features'])
    current_detector.save_models('models/current_anomaly')
    
    # Train humidity anomaly detector
    print("\nTraining humidity anomaly detector...")
    humidity_detector = AnomalyDetector()
    humidity_detector.train_isolation_forest(data['humidity_features'])
    humidity_detector.save_models('models/humidity_anomaly')
    
    # Train vibration anomaly detector
    print("\nTraining vibration anomaly detector...")
    vibration_detector = AnomalyDetector()
    vibration_detector.train_isolation_forest(data['vibration_features'])
    vibration_detector.save_models('models/vibration_anomaly')
    
    # Train pressure anomaly detector
    print("\nTraining pressure anomaly detector...")
    pressure_detector = AnomalyDetector()
    pressure_detector.train_isolation_forest(data['pressure_features'])
    pressure_detector.save_models('models/pressure_anomaly')
    
    # Train viscosity anomaly detector
    print("\nTraining viscosity anomaly detector...")
    viscosity_detector = AnomalyDetector()
    viscosity_detector.train_isolation_forest(data['viscosity_features'])
    viscosity_detector.save_models('models/viscosity_anomaly')
    
    # Train power anomaly detector
    print("\nTraining power anomaly detector...")
    power_detector = AnomalyDetector()
    power_detector.train_isolation_forest(data['power_features'])
    power_detector.save_models('models/power_anomaly')
    
    # Train predictive maintenance model
    print("\nTraining predictive maintenance model...")
    maintenance_model = PredictiveMaintenance()
    maintenance_model.train(data['maintenance_features'], data['maintenance_targets'])
    maintenance_model.save_model('models/maintenance')
    
    # Evaluate models
    print("\nEvaluating models...")
    
    # Temperature anomaly detection
    temp_predictions = temp_detector.detect_anomalies(data['temp_features'])
    temp_accuracy = np.mean(temp_predictions == data['temp_anomalies'])
    print(f"Temperature anomaly detection accuracy: {temp_accuracy:.2f}")
    
    # Current anomaly detection
    current_predictions = current_detector.detect_anomalies(data['current_features'])
    current_accuracy = np.mean(current_predictions == data['current_anomalies'])
    print(f"Current anomaly detection accuracy: {current_accuracy:.2f}")
    
    # Humidity anomaly detection
    humidity_predictions = humidity_detector.detect_anomalies(data['humidity_features'])
    humidity_accuracy = np.mean(humidity_predictions == data['humidity_anomalies'])
    print(f"Humidity anomaly detection accuracy: {humidity_accuracy:.2f}")
    
    # Vibration anomaly detection
    vibration_predictions = vibration_detector.detect_anomalies(data['vibration_features'])
    vibration_accuracy = np.mean(vibration_predictions == data['vibration_anomalies'])
    print(f"Vibration anomaly detection accuracy: {vibration_accuracy:.2f}")
    
    # Pressure anomaly detection
    pressure_predictions = pressure_detector.detect_anomalies(data['pressure_features'])
    pressure_accuracy = np.mean(pressure_predictions == data['pressure_anomalies'])
    print(f"Pressure anomaly detection accuracy: {pressure_accuracy:.2f}")
    
    # Viscosity anomaly detection
    viscosity_predictions = viscosity_detector.detect_anomalies(data['viscosity_features'])
    viscosity_accuracy = np.mean(viscosity_predictions == data['viscosity_anomalies'])
    print(f"Viscosity anomaly detection accuracy: {viscosity_accuracy:.2f}")
    
    # Power anomaly detection
    power_predictions = power_detector.detect_anomalies(data['power_features'])
    power_accuracy = np.mean(power_predictions == data['power_anomalies'])
    print(f"Power anomaly detection accuracy: {power_accuracy:.2f}")
    
    # Predictive maintenance
    maintenance_predictions, maintenance_probs = maintenance_model.predict(data['maintenance_features'])
    maintenance_accuracy = np.mean(maintenance_predictions == data['maintenance_targets'])
    print(f"Predictive maintenance accuracy: {maintenance_accuracy:.2f}")
    
    # Count of predicted anomalies and maintenance (number of 1s in predictions)
    anomaly_counts = {
        "temp_anomalies": int(np.sum(temp_predictions)),
        "temp_maintenance": int(np.sum((maintenance_predictions == 1) & (temp_predictions == 1))),
        "current_anomalies": int(np.sum(current_predictions)),
        "current_maintenance": int(np.sum((maintenance_predictions == 1) & (current_predictions == 1))),
        "humidity_anomalies": int(np.sum(humidity_predictions)),
        "humidity_maintenance": int(np.sum((maintenance_predictions == 1) & (humidity_predictions == 1))),
        "vibration_anomalies": int(np.sum(vibration_predictions)),
        "vibration_maintenance": int(np.sum((maintenance_predictions == 1) & (vibration_predictions == 1))),
        "pressure_anomalies": int(np.sum(pressure_predictions)),
        "pressure_maintenance": int(np.sum((maintenance_predictions == 1) & (pressure_predictions == 1))),
        "viscosity_anomalies": int(np.sum(viscosity_predictions)),
        "viscosity_maintenance": int(np.sum((maintenance_predictions == 1) & (viscosity_predictions == 1))),
        "power_anomalies": int(np.sum(power_predictions)),
        "power_maintenance": int(np.sum((maintenance_predictions == 1) & (power_predictions == 1))),
    }

    # Save both accuracy and counts to JSON
    results = {
        "temp_accuracy": float(temp_accuracy),
        "current_accuracy": float(current_accuracy),
        "humidity_accuracy": float(humidity_accuracy),
        "vibration_accuracy": float(vibration_accuracy),
        "pressure_accuracy": float(pressure_accuracy),
        "viscosity_accuracy": float(viscosity_accuracy),
        "power_accuracy": float(power_accuracy),
        "maintenance_accuracy": float(maintenance_accuracy),
        **anomaly_counts
    }
    with open('models/model_results.json', 'w') as f:
        json.dump(results, f)
    
    novice_counts = {
        "temp_anomalies": anomaly_counts["temp_anomalies"],
        "temp_maintenance": anomaly_counts["temp_maintenance"],
        "current_anomalies": anomaly_counts["current_anomalies"],
        "current_maintenance": anomaly_counts["current_maintenance"],
        "humidity_anomalies": anomaly_counts["humidity_anomalies"],
        "humidity_maintenance": anomaly_counts["humidity_maintenance"],
        "vibration_anomalies": anomaly_counts["vibration_anomalies"],
        "vibration_maintenance": anomaly_counts["vibration_maintenance"],
        "pressure_anomalies": anomaly_counts["pressure_anomalies"],
        "pressure_maintenance": anomaly_counts["pressure_maintenance"],
        "viscosity_anomalies": anomaly_counts["viscosity_anomalies"],
        "viscosity_maintenance": anomaly_counts["viscosity_maintenance"],
        "power_anomalies": anomaly_counts["power_anomalies"],
        "power_maintenance": anomaly_counts["power_maintenance"],
    }
    with open('data/processed/novice_counts.json', 'w') as f:
        json.dump(novice_counts, f)

    print("\nTraining complete!")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train models
    train_models()