import json
import os
from datetime import datetime
from typing import List, Dict, Any

class AnomalyLogger:
    def __init__(self, log_file_path: str = "anomaly_log.json"):
        """
        Initialize the anomaly logger
        
        Args:
            log_file_path (str): Path to the JSON log file
        """
        self.log_file_path = log_file_path
        self.max_log_entries = 1000  # Keep last 1000 anomalies
        
        # Create log file if it doesn't exist
        if not os.path.exists(log_file_path):
            self._create_empty_log()
    
    def _create_empty_log(self):
        """Create an empty log file with proper structure"""
        empty_log = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_anomalies": 0,
                "last_updated": datetime.now().isoformat()
            },
            "anomalies": []
        }
        with open(self.log_file_path, 'w') as f:
            json.dump(empty_log, f, indent=2)
    
    def log_anomaly(self, anomaly_type: str, value: float, unit: str, sensor_data: Dict[str, Any]):
        """
        Log a detected anomaly
        
        Args:
            anomaly_type (str): Type of anomaly (e.g., "TEMPERATURE", "HUMIDITY")
            value (float): The anomalous value
            unit (str): Unit of measurement (e.g., "°C", "%", "A", "W")
            sensor_data (dict): Complete sensor data at time of anomaly
        """
        anomaly_entry = {
            "timestamp": datetime.now().isoformat(),
            "anomaly_type": anomaly_type,
            "value": value,
            "unit": unit,
            "sensor_data": sensor_data,
            "severity": self._determine_severity(anomaly_type, value)
        }
        
        # Load existing log
        try:
            with open(self.log_file_path, 'r') as f:
                log_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._create_empty_log()
            with open(self.log_file_path, 'r') as f:
                log_data = json.load(f)
        
        # Add new anomaly
        log_data["anomalies"].append(anomaly_entry)
        
        # Keep only the last max_log_entries
        if len(log_data["anomalies"]) > self.max_log_entries:
            log_data["anomalies"] = log_data["anomalies"][-self.max_log_entries:]
        
        # Update metadata
        log_data["metadata"]["total_anomalies"] = len(log_data["anomalies"])
        log_data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        # Save updated log
        with open(self.log_file_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def _determine_severity(self, anomaly_type: str, value: float) -> str:
        """Determine severity level based on anomaly type and value"""
        if anomaly_type == "TEMPERATURE":
            if value >= 60 or value <= -5:
                return "CRITICAL"
            elif value >= 45 or value <= 5:
                return "HIGH"
            else:
                return "MEDIUM"
        elif anomaly_type == "HUMIDITY":
            if value >= 98 or value <= 2:
                return "CRITICAL"
            elif value >= 90 or value <= 10:
                return "HIGH"
            else:
                return "MEDIUM"
        elif anomaly_type == "CURRENT":
            if value >= 20 or value <= 0.1:
                return "CRITICAL"
            elif value >= 12 or value <= 1:
                return "HIGH"
            else:
                return "MEDIUM"
        elif anomaly_type == "POWER":
            if value >= 3000 or value <= 100:
                return "CRITICAL"
            elif value >= 2000 or value <= 500:
                return "HIGH"
            else:
                return "MEDIUM"
        else:
            return "MEDIUM"
    
    def get_recent_anomalies(self, limit: int = 50) -> List[Dict]:
        """
        Get recent anomalies from the log
        
        Args:
            limit (int): Maximum number of anomalies to return
            
        Returns:
            List[Dict]: List of recent anomaly entries
        """
        try:
            with open(self.log_file_path, 'r') as f:
                log_data = json.load(f)
            
            # Return the most recent anomalies
            return log_data["anomalies"][-limit:]
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def get_anomaly_stats(self) -> Dict[str, Any]:
        """
        Get statistics about logged anomalies
        
        Returns:
            Dict[str, Any]: Statistics about anomalies
        """
        try:
            with open(self.log_file_path, 'r') as f:
                log_data = json.load(f)
            
            anomalies = log_data["anomalies"]
            
            if not anomalies:
                return {
                    "total_anomalies": 0,
                    "anomaly_types": {},
                    "severity_counts": {},
                    "recent_activity": False
                }
            
            # Count by type
            type_counts = {}
            severity_counts = {}
            
            for anomaly in anomalies:
                anomaly_type = anomaly["anomaly_type"]
                severity = anomaly["severity"]
                
                type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Check for recent activity (last 10 minutes)
            recent_threshold = datetime.now().timestamp() - 600  # 10 minutes ago
            recent_anomalies = [
                a for a in anomalies 
                if datetime.fromisoformat(a["timestamp"]).timestamp() > recent_threshold
            ]
            
            return {
                "total_anomalies": len(anomalies),
                "anomaly_types": type_counts,
                "severity_counts": severity_counts,
                "recent_activity": len(recent_anomalies) > 0,
                "recent_count": len(recent_anomalies)
            }
            
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "total_anomalies": 0,
                "anomaly_types": {},
                "severity_counts": {},
                "recent_activity": False
            }
    
    def clear_log(self):
        """Clear all anomalies from the log"""
        self._create_empty_log() 