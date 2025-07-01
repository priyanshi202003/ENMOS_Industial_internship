#!/usr/bin/env python3
"""
Test script to verify ML anomaly detection is working
"""

import sys
import os
import numpy as np
import pandas as pd
from collections import deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_models.anomaly_detection import AnomalyDetector

def test_ml_detection():
    """Test ML anomaly detection with extreme values"""
    print("🧪 Testing ML Anomaly Detection")
    print("=" * 50)
    
    # Initialize detectors
    models_dir = os.path.join('models')
    temp_detector = AnomalyDetector()
    humidity_detector = AnomalyDetector()
    current_detector = AnomalyDetector()
    power_detector = AnomalyDetector()
    
    # Load models
    print("📥 Loading ML models...")
    try:
        temp_loaded = temp_detector.load_models(os.path.join(models_dir, 'temperature_anomaly'))
        print(f"Temperature model loaded: {temp_loaded}")
        
        humidity_loaded = humidity_detector.load_models(os.path.join(models_dir, 'humidity_anomaly'))
        print(f"Humidity model loaded: {humidity_loaded}")
        
        current_loaded = current_detector.load_models(os.path.join(models_dir, 'current_anomaly'))
        print(f"Current model loaded: {current_loaded}")
        
        power_loaded = power_detector.load_models(os.path.join(models_dir, 'power_anomaly'))
        print(f"Power model loaded: {power_loaded}")
        
        if all([temp_loaded, humidity_loaded, current_loaded, power_loaded]):
            print("✅ All models loaded successfully!")
        else:
            print("❌ Some models failed to load")
            return
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Create feature extraction function (same as in Arduino simulation)
    def create_features_from_window(window_data):
        """Create feature vector from sliding window"""
        if len(window_data) < 24:
            padded_data = list(window_data) + [window_data[-1]] * (24 - len(window_data))
        else:
            padded_data = list(window_data)
        
        data_array = np.array(padded_data)
        
        features = [
            data_array.mean(),
            data_array.std(),
            data_array.max(),
            data_array.min(),
            np.median(data_array),
            float(pd.Series(data_array).skew()),
            float(pd.Series(data_array).kurtosis())
        ]
        
        return np.array(features).reshape(1, -1)
    
    # Test with normal values first
    print("\n🔍 Testing with normal values...")
    normal_temp_window = deque([25, 26, 25, 24, 25, 26, 25, 24, 25, 26, 25, 24, 25, 26, 25, 24, 25, 26, 25, 24, 25, 26, 25, 24])
    normal_humidity_window = deque([60, 61, 60, 59, 60, 61, 60, 59, 60, 61, 60, 59, 60, 61, 60, 59, 60, 61, 60, 59, 60, 61, 60, 59])
    normal_current_window = deque([5, 5.1, 5, 4.9, 5, 5.1, 5, 4.9, 5, 5.1, 5, 4.9, 5, 5.1, 5, 4.9, 5, 5.1, 5, 4.9, 5, 5.1, 5, 4.9])
    normal_power_window = deque([1100, 1122, 1100, 1078, 1100, 1122, 1100, 1078, 1100, 1122, 1100, 1078, 1100, 1122, 1100, 1078, 1100, 1122, 1100, 1078, 1100, 1122, 1100, 1078])
    
    temp_features = create_features_from_window(normal_temp_window)
    humidity_features = create_features_from_window(normal_humidity_window)
    current_features = create_features_from_window(normal_current_window)
    power_features = create_features_from_window(normal_power_window)
    
    temp_normal = temp_detector.detect_anomalies(temp_features, method='isolation_forest')[0]
    humidity_normal = humidity_detector.detect_anomalies(humidity_features, method='isolation_forest')[0]
    current_normal = current_detector.detect_anomalies(current_features, method='isolation_forest')[0]
    power_normal = power_detector.detect_anomalies(power_features, method='isolation_forest')[0]
    
    print(f"   Normal Temperature (25°C): {'ANOMALY' if temp_normal else 'NORMAL'}")
    print(f"   Normal Humidity (60%): {'ANOMALY' if humidity_normal else 'NORMAL'}")
    print(f"   Normal Current (5A): {'ANOMALY' if current_normal else 'NORMAL'}")
    print(f"   Normal Power (1100W): {'ANOMALY' if power_normal else 'NORMAL'}")
    
    # Test with extreme values
    print("\n🔥 Testing with extreme values...")
    
    # Extreme temperature
    extreme_temp_window = deque([25, 26, 25, 24, 25, 26, 25, 24, 25, 26, 25, 24, 25, 26, 25, 24, 25, 26, 25, 24, 25, 26, 25, 80])  # Last value is extreme
    extreme_temp_features = create_features_from_window(extreme_temp_window)
    temp_extreme = temp_detector.detect_anomalies(extreme_temp_features, method='isolation_forest')[0]
    print(f"   Extreme Temperature (80°C): {'ANOMALY' if temp_extreme else 'NORMAL'}")
    
    # Extreme humidity
    extreme_humidity_window = deque([60, 61, 60, 59, 60, 61, 60, 59, 60, 61, 60, 59, 60, 61, 60, 59, 60, 61, 60, 59, 60, 61, 60, 0])  # Last value is extreme
    extreme_humidity_features = create_features_from_window(extreme_humidity_window)
    humidity_extreme = humidity_detector.detect_anomalies(extreme_humidity_features, method='isolation_forest')[0]
    print(f"   Extreme Humidity (0%): {'ANOMALY' if humidity_extreme else 'NORMAL'}")
    
    # Extreme current
    extreme_current_window = deque([5, 5.1, 5, 4.9, 5, 5.1, 5, 4.9, 5, 5.1, 5, 4.9, 5, 5.1, 5, 4.9, 5, 5.1, 5, 4.9, 5, 5.1, 5, 25])  # Last value is extreme
    extreme_current_features = create_features_from_window(extreme_current_window)
    current_extreme = current_detector.detect_anomalies(extreme_current_features, method='isolation_forest')[0]
    print(f"   Extreme Current (25A): {'ANOMALY' if current_extreme else 'NORMAL'}")
    
    # Extreme power
    extreme_power_window = deque([1100, 1122, 1100, 1078, 1100, 1122, 1100, 1078, 1100, 1122, 1100, 1078, 1100, 1122, 1100, 1078, 1100, 1122, 1100, 1078, 1100, 1122, 1100, 5000])  # Last value is extreme
    extreme_power_features = create_features_from_window(extreme_power_window)
    power_extreme = power_detector.detect_anomalies(extreme_power_features, method='isolation_forest')[0]
    print(f"   Extreme Power (5000W): {'ANOMALY' if power_extreme else 'NORMAL'}")
    
    print("\n✅ ML detection test completed!")

if __name__ == "__main__":
    test_ml_detection() 