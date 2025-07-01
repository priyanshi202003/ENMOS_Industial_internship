import numpy as np
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.lstm_model = None
        self.is_trained = False

    def create_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.lstm_model = model
        return model

    def train_isolation_forest(self, data):
        """Train Isolation Forest model on historical data"""
        self.isolation_forest.fit(data)
        self.is_trained = True

    def train_lstm(self, X_train, y_train, epochs=50, batch_size=32):
        """Train LSTM model for sequence-based anomaly detection"""
        if self.lstm_model is None:
            self.create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        return history

    def detect_anomalies(self, data, method='isolation_forest'):
        """Detect anomalies using either Isolation Forest or LSTM"""
        if not self.is_trained:
            raise ValueError("Model needs to be trained before detection")

        if method == 'isolation_forest':
            predictions = self.isolation_forest.predict(data)
            return predictions == -1  # -1 indicates anomaly
        elif method == 'lstm':
            predictions = self.lstm_model.predict(data)
            return predictions > 0.5  # Threshold for anomaly
        else:
            raise ValueError("Invalid method specified")

    def save_models(self, path_prefix):
        """Save trained models"""
        if self.is_trained:
            joblib.dump(self.isolation_forest, f"{path_prefix}_isolation_forest.joblib")
            if self.lstm_model:
                self.lstm_model.save(f"{path_prefix}_lstm_model.h5")

    def load_models(self, path_prefix):
        """Load trained models"""
        try:
            self.isolation_forest = joblib.load(f"{path_prefix}_isolation_forest.joblib")
            self.is_trained = True
            return True
        except:
            return False 