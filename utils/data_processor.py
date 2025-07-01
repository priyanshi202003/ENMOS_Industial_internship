import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def prepare_time_series_data(data, window_size=24):
    """
    Prepare time series data for ML models
    
    Args:
        data (pd.DataFrame): Input data with timestamp and values
        window_size (int): Size of the sliding window for feature extraction
    
    Returns:
        np.array: Feature matrix with shape (n_samples, n_features)
    """
    # Ensure data is sorted by timestamp
    data = data.sort_values('timestamp')
    
    # Create features using sliding window
    features = []
    
    # Ensure we have enough data points
    if len(data) <= window_size:
        raise ValueError(f"Not enough data points. Need at least {window_size + 1} points, got {len(data)}")
    
    for i in range(len(data) - window_size):
        window = data.iloc[i:i+window_size]
        
        # Extract features from window
        window_features = [
            window['value'].mean(),
            window['value'].std(),
            window['value'].max(),
            window['value'].min(),
            window['value'].median(),
            window['value'].skew(),
            window['value'].kurtosis()
        ]
        
        features.append(window_features)
    
    # Convert to numpy array and ensure it's 2D
    features_array = np.array(features)
    if len(features_array.shape) == 1:
        features_array = features_array.reshape(-1, 1)
    
    return features_array

def detect_seasonality(data, period_range=range(2, 25)):
    """
    Detect seasonality in time series data
    
    Args:
        data (pd.Series): Time series data
        period_range (range): Range of periods to check for seasonality
    
    Returns:
        int: Detected seasonality period
    """
    # Calculate autocorrelation
    autocorr = pd.Series(data).autocorr()
    
    # Find period with maximum autocorrelation
    max_corr = -1
    best_period = None
    
    for period in period_range:
        corr = pd.Series(data).autocorr(lag=period)
        if corr > max_corr:
            max_corr = corr
            best_period = period
    
    return best_period

def calculate_energy_metrics(current_data, voltage=220):
    """
    Calculate energy consumption metrics
    
    Args:
        current_data (pd.Series): Current measurements in amperes
        voltage (float): Voltage in volts (default: 220V)
    
    Returns:
        dict: Dictionary containing energy metrics
    """
    # Calculate power (P = V * I)
    power = current_data * voltage
    
    # Calculate energy consumption (kWh)
    energy = power.sum() / 1000  # Convert to kWh
    
    # Calculate peak power
    peak_power = power.max()
    
    # Calculate average power
    avg_power = power.mean()
    
    return {
        'total_energy': energy,
        'peak_power': peak_power,
        'average_power': avg_power
    }

def generate_anomaly_labels(data, threshold_std=3):
    """
    Generate anomaly labels based on statistical threshold
    
    Args:
        data (pd.Series): Input data
        threshold_std (float): Number of standard deviations for threshold
    
    Returns:
        np.array: Binary array indicating anomalies (1) and normal points (0)
    """
    mean = data.mean()
    std = data.std()
    threshold = threshold_std * std
    
    return np.array([1 if abs(x - mean) > threshold else 0 for x in data]) 