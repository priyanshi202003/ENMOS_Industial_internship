import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class PredictiveMaintenance:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_features(self, data):
        """Prepare features for the model"""
        # Extract relevant features from time series data
        features = []
        for i in range(len(data) - 24):  # Use 24 time steps for prediction
            window = data[i:i+24]
            features.append([
                np.mean(window),  # Mean
                np.std(window),   # Standard deviation
                np.max(window),   # Maximum
                np.min(window),   # Minimum
                np.ptp(window),   # Peak to peak
                np.median(window) # Median
            ])
        return np.array(features)

    def train(self, X, y):
        """Train the predictive maintenance model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, X):
        """Make predictions for maintenance needs"""
        if not self.is_trained:
            raise ValueError("Model needs to be trained before prediction")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities

    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model needs to be trained before getting feature importance")
        
        return self.model.feature_importances_

    def save_model(self, path_prefix):
        """Save trained model and scaler"""
        if self.is_trained:
            joblib.dump(self.model, f"{path_prefix}_model.joblib")
            joblib.dump(self.scaler, f"{path_prefix}_scaler.joblib")

    def load_model(self, path_prefix):
        """Load trained model and scaler"""
        try:
            self.model = joblib.load(f"{path_prefix}_model.joblib")
            self.scaler = joblib.load(f"{path_prefix}_scaler.joblib")
            self.is_trained = True
            return True
        except:
            return False 