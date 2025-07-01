import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import requests
import json
import threading
import time
import pathlib
from collections import deque
from dotenv import load_dotenv
import yagmail  # Add this import

load_dotenv()  # Load .env file

EMAIL_USER = os.getenv("GMAIL_USER")
EMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")

# Your Gmail password (⚠️ Not recommended for production)
ALERT_TO_EMAIL = "jameelasaeendran@gmail.com"  # Who receives the alerts

def send_alert_email(subject, body):
    try:
        yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASSWORD)
        yag.send(
            to=ALERT_TO_EMAIL,
            subject=subject,
            contents=body
        )
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
    finally:
        if 'yag' in locals():
            yag.close()

# Add parent directory to path to import ML models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_models.anomaly_detection import AnomalyDetector
from ml_models.predictive_maintenance import PredictiveMaintenance
from utils.anomaly_logger import AnomalyLogger

# Initialize anomaly logger
anomaly_logger = AnomalyLogger("anomaly_log.json")

# Clear anomaly log when web app starts
print("🧹 Clearing anomaly log on startup...")
anomaly_logger.clear_log()
print("✅ Anomaly log cleared - starting fresh!")

# Track app startup time
APP_START_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Arduino Integration Configuration
ARDUINO_SERVER_URL = "http://localhost:5001"
arduino_data_cache = []
arduino_connection_status = "disconnected"

# Path to the simulated Arduino data file
SIMULATED_ARDUINO_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'arduino_integration', 'latest_arduino_data.json')

# Store rolling window for live graphs
sim_data_history = {
    'temperature': deque(maxlen=100),
    'humidity': deque(maxlen=100),
    'voltage': deque(maxlen=100),
    'current': deque(maxlen=100),
    'time': deque(maxlen=100)
}

# Track last cleanup time
last_cleanup_time = datetime.now()

def cleanup_data_history():
    """Periodically cleanup data history to prevent memory issues"""
    global last_cleanup_time, sim_data_history
    now = datetime.now()
    
    # Cleanup every 10 minutes
    if (now - last_cleanup_time).total_seconds() > 600:
        try:
            # Keep only the last 50 points to reduce memory usage
            for key in sim_data_history:
                if len(sim_data_history[key]) > 50:
                    # Convert to list, take last 50, convert back to deque
                    temp_list = list(sim_data_history[key])
                    sim_data_history[key] = deque(temp_list[-50:], maxlen=100)
            
            last_cleanup_time = now
            print("🧹 Data history cleaned up")
        except Exception as e:
            print(f"⚠️  Error during data cleanup: {e}")

def fetch_arduino_data():
    """Fetch data from Arduino receiver"""
    global arduino_data_cache, arduino_connection_status
    try:
        response = requests.get(f"{ARDUINO_SERVER_URL}/api/arduino/data?limit=100", timeout=2)
        if response.status_code == 200:
            data = response.json()
            arduino_data_cache = data
            arduino_connection_status = "connected"
            return data
        else:
            arduino_connection_status = "error"
            return []
    except:
        arduino_connection_status = "disconnected"
        return []

def get_arduino_status():
    """Get Arduino connection status"""
    global arduino_connection_status
    try:
        response = requests.get(f"{ARDUINO_SERVER_URL}/api/arduino/status", timeout=2)
        if response.status_code == 200:
            status = response.json()
            arduino_connection_status = "connected"
            return status
        else:
            arduino_connection_status = "error"
            return None
    except:
        arduino_connection_status = "disconnected"
        return None

def read_simulated_arduino_data():
    """Read the latest simulated Arduino data from file with improved error handling."""
    try:
        if not os.path.exists(SIMULATED_ARDUINO_DATA_PATH):
            print(f"⚠️  Simulated Arduino data file not found: {SIMULATED_ARDUINO_DATA_PATH}")
            return None
        
        with open(SIMULATED_ARDUINO_DATA_PATH, 'r') as f:
            data = json.load(f)
        
        # Validate data structure
        required_keys = ['temperature', 'humidity', 'voltage', 'current']
        if not all(key in data for key in required_keys):
            print(f"⚠️  Invalid data structure in {SIMULATED_ARDUINO_DATA_PATH}")
            return None
            
        return data
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON decode error reading Arduino data: {e}")
        return None
    except FileNotFoundError:
        print(f"⚠️  Arduino data file not found: {SIMULATED_ARDUINO_DATA_PATH}")
        return None
    except Exception as e:
        print(f"⚠️  Error reading simulated Arduino data: {e}")
        return None

def get_anomaly_log_data():
    """Get anomaly log data for dashboard display"""
    try:
        return anomaly_logger.get_recent_anomalies(100)  # Get last 100 anomalies
    except Exception as e:
        print(f"Error reading anomaly log: {e}")
        return []

def get_anomaly_stats():
    """Get anomaly statistics for dashboard display"""
    try:
        return anomaly_logger.get_anomaly_stats()
    except Exception as e:
        print(f"Error reading anomaly stats: {e}")
        return {
            "total_anomalies": 0,
            "anomaly_types": {},
            "severity_counts": {},
            "recent_activity": False
        }

# Load the real combined data
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'combined_data.csv')
combined_df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])

# Initialize ML models
temp_detector = AnomalyDetector()
current_detector = AnomalyDetector()
humidity_detector = AnomalyDetector()
vibration_detector = AnomalyDetector()
pressure_detector = AnomalyDetector()
viscosity_detector = AnomalyDetector()
power_detector = AnomalyDetector()
maintenance_model = PredictiveMaintenance()

# Load trained models
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
temp_detector.load_models(os.path.join(models_dir, 'temperature_anomaly'))
current_detector.load_models(os.path.join(models_dir, 'current_anomaly'))
humidity_detector.load_models(os.path.join(models_dir, 'humidity_anomaly'))
vibration_detector.load_models(os.path.join(models_dir, 'vibration_anomaly'))
pressure_detector.load_models(os.path.join(models_dir, 'pressure_anomaly'))
viscosity_detector.load_models(os.path.join(models_dir, 'viscosity_anomaly'))
power_detector.load_models(os.path.join(models_dir, 'power_anomaly'))
maintenance_model.load_model(os.path.join(models_dir, 'maintenance'))

# Initialize Dash app with dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Configure plotly to use local assets and improve loading
app.config.suppress_callback_exceptions = True
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

# Custom CSS for better dark mode styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>ENMOS Monitoring System - Arduino Integration</title>
        {%favicon%}
        {%css%}
        <!-- Plotly.js with fallback -->
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>
            // Fallback if CDN fails
            if (typeof Plotly === 'undefined') {
                document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.24.1/plotly.min.js"><\/script>');
            }
        </script>
        <style>
            :root {
                --electric-green: #39ff14;
                --peacock-blue: #003844;
                --card-bg: #1a1a1a;
                --card-border: #39ff14;
                --tab-bg: #003844;
                --tab-active-bg: #39ff14;
                --tab-active-color: #003844;
                --tab-inactive-color: #39ff14;
            }
            body {
                background-color: var(--card-bg) !important;
                color: #ffffff !important;
            }
            .navbar {
                background: linear-gradient(90deg, #003844 0%, #0093af 100%) !important;
                border-bottom: 2.5px solid #39ff14;
                box-shadow: 0 4px 24px 0 rgba(0,255,100,0.10);
            }
            .navbar-brand span {
                color: #39ff14 !important;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
            .card, .tab-content, .tab-pane, .bg-light {
                background-color: var(--card-bg) !important;
                border: 2px solid var(--peacock-blue) !important;
                color: #ffffff !important;
                border-radius: 18px;
                box-shadow: 0 4px 24px 0 rgba(0,147,175,0.10);
            }
            .card-body {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            .nav-tabs .nav-link {
                color: var(--tab-inactive-color) !important;
                background: var(--tab-bg) !important;
                border: none !important;
                font-weight: 700;
                font-size: 1.1rem;
                letter-spacing: 1px;
                margin-right: 4px;
                border-radius: 12px 12px 0 0 !important;
                transition: background 0.2s, color 0.2s;
            }
            .nav-tabs .nav-link.active, .nav-tabs .nav-link:focus, .nav-tabs .nav-link:hover {
                color: var(--tab-active-color) !important;
                background: var(--tab-active-bg) !important;
                border: none !important;
                box-shadow: 0 2px 12px 0 rgba(57,255,20,0.18);
            }
            .fw-semibold {
                color: var(--electric-green);
            }
            .text-muted { color: #b2f7ef !important; }
            .text-danger { color: #ff5e5e !important; }
            .text-success { color: var(--electric-green) !important; }
            .border-0 { border: none !important; }
            .shadow-sm { box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.3) !important; }
            .btn-enmos {
                background: linear-gradient(90deg, var(--electric-green) 0%, var(--peacock-blue) 100%);
                color: #232526;
                font-weight: 700;
                border: none;
                border-radius: 30px;
                padding: 14px 38px;
                font-size: 1.2rem;
                box-shadow: 0 4px 14px rgba(57,255,20,0.15);
                transition: background 0.2s, color 0.2s, box-shadow 0.2s;
            }
            .btn-enmos:hover {
                background: linear-gradient(90deg, var(--peacock-blue) 0%, var(--electric-green) 100%);
                color: #fff;
                box-shadow: 0 8px 24px rgba(0,147,175,0.25);
            }
            .arduino-status {
                padding: 10px;
                border-radius: 8px;
                margin: 10px 0;
                font-weight: bold;
            }
            .arduino-connected {
                background-color: rgba(57, 255, 20, 0.1);
                border: 2px solid #39ff14;
                color: #39ff14;
            }
            .arduino-disconnected {
                background-color: rgba(255, 94, 94, 0.1);
                border: 2px solid #ff5e5e;
                color: #ff5e5e;
            }
            .sensor-value {
                font-size: 2rem;
                font-weight: bold;
                color: #39ff14;
                margin: 10px 0;
            }
            .sensor-unit {
                font-size: 1rem;
                color: #b2f7ef;
                margin-bottom: 10px;
            }
            .novice-card {
                background: linear-gradient(135deg, #1a1a1a 0%, #2d3748 100%);
                border: 2px solid #39ff14;
                transition: all 0.3s ease;
            }
            .novice-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(57,255,20,0.2);
            }
            .ml-anomaly-card {
                background: linear-gradient(135deg, #003844 0%, #232526 100%);
                border: 2px solid #39ff14;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(255, 71, 87, 0.7); }
                70% { box-shadow: 0 0 0 10px rgba(255, 71, 87, 0); }
                100% { box-shadow: 0 0 0 0 rgba(255, 71, 87, 0); }
            }
            .anomaly-alert {
                background: #111; /* solid black background */
                border: none;
                color: #fff; /* white text */
                padding: 14px 28px;
                border-radius: 10px;
                font-size: 1.08rem;
                font-family: 'Inter', 'Segoe UI', 'Arial', sans-serif;
                font-weight: 500;
                box-shadow: 0 2px 12px 0 rgba(231, 76, 60, 0.08);
                margin-bottom: 1.5rem;
                letter-spacing: 0.2px;
                transition: background 0.2s, color 0.2s;
                display: block;
                text-align: center;
            }
            .anomaly-alert .icon {
                display: none; /* Hide icon for minimalism */
            }
            .anomaly-alert .details {
                font-size: 0.97rem;
                color: #bbb; /* subtle gray for details */
                margin-left: 1em;
                font-weight: 400;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout
app.layout = dbc.Container([
    # Space below navbar
    html.Div(style={"height": "20px"}),

    # Add the interval here so it is always present
    dcc.Interval(
        id='interval-component',
        interval=2*1000,  # Update every 2 seconds
        n_intervals=0
    ),

    # Arduino Status Section
    

    # Real-time ML Anomaly Alert Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("Real-time ML Anomaly Detection", className="text-center mb-3 fw-semibold", style={"color": "#fff", "letterSpacing": "1px"}),
                    html.Div(id='ml-anomaly-alert', className="anomaly-alert mx-auto", style={"textAlign": "center", "maxWidth": "480px"}),
                ])
            ], className="ml-anomaly-card shadow-sm border-0")
        ])
    ], className="mb-4"),

    # Arduino Real-time Data Section
    dbc.Row([
        dbc.Col([
            html.H2("Arduino Real-time Data", className="text-center mb-3 fw-light", style={"color": "#39ff14", "letterSpacing": "2px"}),
            html.Hr(style={"borderTop": "2px solid #39ff14", "width": "180px", "margin": "0 auto 24px auto"}),
            
            # Real-time Sensor Values
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="bi bi-thermometer-half", style={"fontSize": "2rem", "color": "#ff6b6b"}),
                                html.Div("Temperature", className="fw-semibold mt-2 mb-1"),
                                html.Div(id='arduino-temp-value', className="sensor-value"),
                                html.Div("°C", className="sensor-unit"),
                                html.Div(id='arduino-temp-anomaly', className="text-danger", style={"fontSize": "0.8rem"})
                            ], className="text-center")
                        ])
                    ], className="shadow-sm border-0 bg-dark")
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="bi bi-droplet", style={"fontSize": "2rem", "color": "#48dbfb"}),
                                html.Div("Humidity", className="fw-semibold mt-2 mb-1"),
                                html.Div(id='arduino-humidity-value', className="sensor-value"),
                                html.Div("%", className="sensor-unit"),
                                html.Div(id='arduino-humidity-anomaly', className="text-danger", style={"fontSize": "0.8rem"})
                            ], className="text-center")
                        ])
                    ], className="shadow-sm border-0 bg-dark")
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="bi bi-lightning-charge", style={"fontSize": "2rem", "color": "#feca57"}),
                                html.Div("Current", className="fw-semibold mt-2 mb-1"),
                                html.Div(id='arduino-current-value', className="sensor-value"),
                                html.Div("A", className="sensor-unit"),
                                html.Div(id='arduino-current-anomaly', className="text-danger", style={"fontSize": "0.8rem"})
                            ], className="text-center")
                        ])
                    ], className="shadow-sm border-0 bg-dark")
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="bi bi-vibrate", style={"fontSize": "2rem", "color": "#ff9ff3"}),
                                html.Div("Vibration", className="fw-semibold mt-2 mb-1"),
                                html.Div(id='arduino-vibration-value', className="sensor-value"),
                                html.Div("g", className="sensor-unit"),
                                html.Div(id='arduino-vibration-anomaly', className="text-danger", style={"fontSize": "0.8rem"})
                            ], className="text-center")
                        ])
                    ], className="shadow-sm border-0 bg-dark")
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="bi bi-speedometer2", style={"fontSize": "2rem", "color": "#54a0ff"}),
                                html.Div("Pressure", className="fw-semibold mt-2 mb-1"),
                                html.Div(id='arduino-pressure-value', className="sensor-value"),
                                html.Div("hPa", className="sensor-unit"),
                                html.Div(id='arduino-pressure-anomaly', className="text-danger", style={"fontSize": "0.8rem"})
                            ], className="text-center")
                        ])
                    ], className="shadow-sm border-0 bg-dark")
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="bi bi-lightning", style={"fontSize": "2rem", "color": "#5f27cd"}),
                                html.Div("Power", className="fw-semibold mt-2 mb-1"),
                                html.Div(id='arduino-power-value', className="sensor-value"),
                                html.Div("W", className="sensor-unit"),
                                html.Div(id='arduino-power-anomaly', className="text-danger", style={"fontSize": "0.8rem"})
                            ], className="text-center")
                        ])
                    ], className="shadow-sm border-0 bg-dark")
                ], width=2),
            ], className="mb-4"),
            
            # Arduino Data Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Recent Arduino Data", className="text-center mb-3"),
                            html.Div(id='arduino-data-table')
                        ])
                    ], className="shadow-sm border-0 bg-dark")
                ])
            ]),
            
            # Arduino Live Graphs
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Temperature (Live)", className="text-center mb-2"),
                            dcc.Graph(id='sim-temp-graph', style={'height': '250px'})
                        ])
                    ], className="shadow-sm border-0 bg-dark")
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Humidity (Live)", className="text-center mb-2"),
                            dcc.Graph(id='sim-humidity-graph', style={'height': '250px'})
                        ])
                    ], className="shadow-sm border-0 bg-dark")
                ], width=6),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Voltage (Live)", className="text-center mb-2"),
                            dcc.Graph(id='sim-voltage-graph', style={'height': '250px'})
                        ])
                    ], className="shadow-sm border-0 bg-dark")
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Current (Live)", className="text-center mb-2"),
                            dcc.Graph(id='sim-current-graph', style={'height': '250px'})
                        ])
                    ], className="shadow-sm border-0 bg-dark")
                ], width=6),
            ], className="mb-4"),
            
        ])
    ], className="mb-4"),

    # Non-Technical Section (was Novice)
    dbc.Row([
        dbc.Col([
            html.H2("Non-Technical View", className="text-center mb-3 fw-light", style={"color": "#baff39", "letterSpacing": "2px"}),
            html.Hr(style={"borderTop": "2px solid #39ff14", "width": "180px", "margin": "0 auto 24px auto"}),
            html.Div([
                dbc.Row([
                    # Temperature Card
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="bi bi-thermometer-half", style={"fontSize": "2rem", "color": "#63b3ed"}),
                                    html.Div("Temperature", className="fw-semibold mt-2 mb-1", style={"fontSize": "1rem", "color": "#ffffff"}),
                                    html.Div(id='temp-anomalies', className="text-muted", style={"fontSize": "0.9rem"}),
                                    html.Div(id='temp-maintenance', className="text-danger", style={"fontSize": "0.9rem"})
                                ], className="text-center")
                            ])
                        ], className="novice-card shadow-sm border-0 bg-dark mx-1"),
                        style={"minWidth": "170px", "maxWidth": "200px"}
                    ),
                    # Current Card
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="bi bi-lightning-charge", style={"fontSize": "2rem", "color": "#f6e05e"}),
                                    html.Div("Current", className="fw-semibold mt-2 mb-1", style={"fontSize": "1rem", "color": "#ffffff"}),
                                    html.Div(id='current-anomalies', className="text-muted", style={"fontSize": "0.9rem"}),
                                    html.Div(id='current-maintenance', className="text-danger", style={"fontSize": "0.9rem"})
                                ], className="text-center")
                            ])
                        ], className="novice-card shadow-sm border-0 bg-dark mx-1"),
                        style={"minWidth": "170px", "maxWidth": "200px"}
                    ),
                    # Humidity Card
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="bi bi-droplet", style={"fontSize": "2rem", "color": "#63b3ed"}),
                                    html.Div("Humidity", className="fw-semibold mt-2 mb-1", style={"fontSize": "1rem", "color": "#ffffff"}),
                                    html.Div(id='humidity-anomalies', className="text-muted", style={"fontSize": "0.9rem"}),
                                    html.Div(id='humidity-maintenance', className="text-danger", style={"fontSize": "0.9rem"})
                                ], className="text-center")
                            ])
                        ], className="novice-card shadow-sm border-0 bg-dark mx-1"),
                        style={"minWidth": "170px", "maxWidth": "200px"}
                    ),
                    # Vibration Card
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="bi bi-activity", style={"fontSize": "2rem", "color": "#68d391"}),
                                    html.Div("Vibration", className="fw-semibold mt-2 mb-1", style={"fontSize": "1rem", "color": "#ffffff"}),
                                    html.Div(id='vibration-anomalies', className="text-muted", style={"fontSize": "0.9rem"}),
                                    html.Div(id='vibration-maintenance', className="text-danger", style={"fontSize": "0.9rem"})
                                ], className="text-center")
                            ])
                        ], className="novice-card shadow-sm border-0 bg-dark mx-1"),
                        style={"minWidth": "170px", "maxWidth": "200px"}
                    ),
                    # Pressure Card
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="bi bi-speedometer2", style={"fontSize": "2rem", "color": "#fc8181"}),
                                    html.Div("Pressure", className="fw-semibold mt-2 mb-1", style={"fontSize": "1rem", "color": "#ffffff"}),
                                    html.Div(id='pressure-anomalies', className="text-muted", style={"fontSize": "0.9rem"}),
                                    html.Div(id='pressure-maintenance', className="text-danger", style={"fontSize": "0.9rem"})
                                ], className="text-center")
                            ])
                        ], className="novice-card shadow-sm border-0 bg-dark mx-1"),
                        style={"minWidth": "170px", "maxWidth": "200px"}
                    ),
                    # Viscosity Card
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="bi bi-droplet-half", style={"fontSize": "2rem", "color": "#9f7aea"}),
                                    html.Div("Viscosity", className="fw-semibold mt-2 mb-1", style={"fontSize": "1rem", "color": "#ffffff"}),
                                    html.Div(id='viscosity-anomalies', className="text-muted", style={"fontSize": "0.9rem"}),
                                    html.Div(id='viscosity-maintenance', className="text-danger", style={"fontSize": "0.9rem"})
                                ], className="text-center")
                            ])
                        ], className="novice-card shadow-sm border-0 bg-dark mx-1"),
                        style={"minWidth": "170px", "maxWidth": "200px"}
                    ),
                    # Power Card
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="bi bi-lightbulb", style={"fontSize": "2rem", "color": "#f6e05e"}),
                                    html.Div("Power", className="fw-semibold mt-2 mb-1", style={"fontSize": "1rem", "color": "#ffffff"}),
                                    html.Div(id='power-anomalies', className="text-muted", style={"fontSize": "0.9rem"}),
                                    html.Div(id='power-maintenance', className="text-danger", style={"fontSize": "0.9rem"})
                                ], className="text-center")
                            ])
                        ], className="novice-card shadow-sm border-0 bg-dark mx-1"),
                        style={"minWidth": "170px", "maxWidth": "200px"}
                    ),
                ],
                className="flex-nowrap justify-content-center g-2",
                style={
                    "overflowX": "auto",
                    "padding": "18px 0",
                    "background": "linear-gradient(90deg, #003844 0%, #0093af22 100%)",
                    "borderRadius": "18px",
                    "boxShadow": "0 4px 24px 0 rgba(0,147,175,0.10)"
                }
                )
            ])
        ], width=12)
    ], className="mt-4 mb-5"),
    
    # Technical Section
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab([
                    html.Div([
                        html.H3("Real-time Monitoring", className="text-center mb-4 fw-semibold", style={"color": "#39ff14", "letterSpacing": "1px"}),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='live-temperature', config={"displayModeBar": False}), width=6),
                            dbc.Col(dcc.Graph(id='live-current', config={"displayModeBar": False}), width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='live-humidity', config={"displayModeBar": False}), width=6),
                            dbc.Col(dcc.Graph(id='live-vibration', config={"displayModeBar": False}), width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='live-pressure', config={"displayModeBar": False}), width=6),
                            dbc.Col(dcc.Graph(id='live-viscosity', config={"displayModeBar": False}), width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='live-power', config={"displayModeBar": False}), width=12),
                        ]),
                    ], style={"background": "#003844", "borderRadius": "18px", "padding": "24px 12px", "boxShadow": "0 4px 24px 0 rgba(57,255,20,0.10)"})
                ], label="Real-time Monitoring", tab_id="tab-rt"),
                dbc.Tab([
                    html.Div([
                        html.H3("Anomaly Detection", className="text-center mb-4 fw-semibold", style={"color": "#39ff14", "letterSpacing": "1px"}),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='anomaly-plot-temp', config={"displayModeBar": False}), width=6),
                            dbc.Col(dcc.Graph(id='anomaly-plot-current', config={"displayModeBar": False}), width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='anomaly-plot-humidity', config={"displayModeBar": False}), width=6),
                            dbc.Col(dcc.Graph(id='anomaly-plot-vibration', config={"displayModeBar": False}), width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='anomaly-plot-pressure', config={"displayModeBar": False}), width=6),
                            dbc.Col(dcc.Graph(id='anomaly-plot-viscosity', config={"displayModeBar": False}), width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='anomaly-plot-power', config={"displayModeBar": False}), width=12),
                        ])
                    ], style={"background": "#003844", "borderRadius": "18px", "padding": "24px 12px", "boxShadow": "0 4px 24px 0 rgba(57,255,20,0.10)"})
                ], label="Anomaly Detection", tab_id="tab-anom"),
                dbc.Tab([
                    html.Div([
                        html.H3("Predictive Maintenance", className="text-center mb-4 fw-semibold", style={"color": "#39ff14", "letterSpacing": "1px"}),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='maintenance-plot-temp', config={"displayModeBar": False}), width=6),
                            dbc.Col(dcc.Graph(id='maintenance-plot-current', config={"displayModeBar": False}), width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='maintenance-plot-humidity', config={"displayModeBar": False}), width=6),
                            dbc.Col(dcc.Graph(id='maintenance-plot-vibration', config={"displayModeBar": False}), width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='maintenance-plot-pressure', config={"displayModeBar": False}), width=6),
                            dbc.Col(dcc.Graph(id='maintenance-plot-viscosity', config={"displayModeBar": False}), width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='maintenance-plot-power', config={"displayModeBar": False}), width=12),
                        ])
                    ], style={"background": "#003844", "borderRadius": "18px", "padding": "24px 12px", "boxShadow": "0 4px 24px 0 rgba(57,255,20,0.10)"})
                ], label="Predictive Maintenance", tab_id="tab-maint"),
                dbc.Tab([
                    html.Div([
                        html.H3("Energy Consumption Analysis", className="text-center mb-4 fw-semibold", style={"color": "#39ff14", "letterSpacing": "1px"}),
                        dcc.Graph(id='energy-plot', config={"displayModeBar": False}),
                        html.Div(id='energy-insights', className="mt-3", style={"color": "#ffffff"})
                    ], style={"background": "#003844", "borderRadius": "18px", "padding": "24px 12px", "boxShadow": "0 4px 24px 0 rgba(57,255,20,0.10)"})
                ], label="Energy Analysis", tab_id="tab-energy"),
                dbc.Tab([
                    html.Div([
                        html.H3("Model Performance - Confusion Matrices", className="text-center mb-4 fw-semibold", style={"color": "#39ff14", "letterSpacing": "1px"}),
                        html.P("Confusion matrices showing the performance of anomaly detection models for each parameter:", 
                               className="text-center mb-4", style={"color": "#b2f7ef"}),
                        dbc.Row([
                            dbc.Col([
                                html.H5("Temperature", className="text-center mb-2", style={"color": "#39ff14"}),
                                html.Img(src="/assets/confusion_matrix_temperature.png", 
                                        style={"width": "100%", "maxWidth": "400px", "borderRadius": "12px"})
                            ], width=6, className="mb-4"),
                            dbc.Col([
                                html.H5("Current", className="text-center mb-2", style={"color": "#39ff14"}),
                                html.Img(src="/assets/confusion_matrix_current.png", 
                                        style={"width": "100%", "maxWidth": "400px", "borderRadius": "12px"})
                            ], width=6, className="mb-4"),
                        ], className="mb-4"),
                        dbc.Row([
                            dbc.Col([
                                html.H5("Humidity", className="text-center mb-2", style={"color": "#39ff14"}),
                                html.Img(src="/assets/confusion_matrix_humidity.png", 
                                        style={"width": "100%", "maxWidth": "400px", "borderRadius": "12px"})
                            ], width=6, className="mb-4"),
                            dbc.Col([
                                html.H5("Vibration", className="text-center mb-2", style={"color": "#39ff14"}),
                                html.Img(src="/assets/confusion_matrix_vibration.png", 
                                        style={"width": "100%", "maxWidth": "400px", "borderRadius": "12px"})
                            ], width=6, className="mb-4"),
                        ], className="mb-4"),
                        dbc.Row([
                            dbc.Col([
                                html.H5("Pressure", className="text-center mb-2", style={"color": "#39ff14"}),
                                html.Img(src="/assets/confusion_matrix_pressure.png", 
                                        style={"width": "100%", "maxWidth": "400px", "borderRadius": "12px"})
                            ], width=6, className="mb-4"),
                            dbc.Col([
                                html.H5("Viscosity", className="text-center mb-2", style={"color": "#39ff14"}),
                                html.Img(src="/assets/confusion_matrix_viscosity.png", 
                                        style={"width": "100%", "maxWidth": "400px", "borderRadius": "12px"})
                            ], width=6, className="mb-4"),
                        ], className="mb-4"),
                        dbc.Row([
                            dbc.Col([
                                html.H5("Power", className="text-center mb-2", style={"color": "#39ff14"}),
                                html.Img(src="/assets/confusion_matrix_power.png", 
                                        style={"width": "100%", "maxWidth": "400px", "borderRadius": "12px"})
                            ], width=6, className="mb-4"),
                            dbc.Col([
                                html.H5("Summary", className="text-center mb-2", style={"color": "#39ff14"}),
                                html.Img(src="/assets/confusion_matrices_summary.png", 
                                        style={"width": "100%", "maxWidth": "400px", "borderRadius": "12px"})
                            ], width=6, className="mb-4"),
                        ], className="mb-4"),
                        html.Div([
                            html.H6("Confusion Matrix Legend:", className="mb-2", style={"color": "#39ff14"}),
                            html.Ul([
                                html.Li("True Negatives (TN): Correctly identified normal data", style={"color": "#ffffff"}),
                                html.Li("False Positives (FP): Incorrectly flagged normal as anomaly", style={"color": "#ffffff"}),
                                html.Li("False Negatives (FN): Missed actual anomalies", style={"color": "#ffffff"}),
                                html.Li("True Positives (TP): Correctly identified anomalies", style={"color": "#ffffff"})
                            ], style={"color": "#b2f7ef"})
                        ], className="mt-4 p-3", style={"background": "#1a1a1a", "borderRadius": "12px", "border": "1px solid #39ff14"})
                    ], style={"background": "#003844", "borderRadius": "18px", "padding": "24px 12px", "boxShadow": "0 4px 24px 0 rgba(57,255,20,0.10)"})
                ], label="Model Performance", tab_id="tab-performance"),
                dbc.Tab([
                    html.Div([
                        html.H3("Anomaly Log", className="text-center mb-4 fw-semibold", style={"color": "#39ff14", "letterSpacing": "1px"}),
                        
                        # Anomaly Statistics
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H5("📊 Anomaly Statistics", className="text-center mb-3", style={"color": "#39ff14"}),
                                        html.Div(id='anomaly-stats-display', className="text-center")
                                    ])
                                ], className="shadow-sm border-0 bg-dark")
                            ], width=12, className="mb-4")
                        ]),
                        
                        # Anomaly Log Table
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H5("📋 Recent Anomalies", className="text-center mb-3", style={"color": "#39ff14"}),
                                        html.Div(id='anomaly-log-table', className="text-center")
                                    ])
                                ], className="shadow-sm border-0 bg-dark")
                            ], width=12)
                        ])
                    ], style={"background": "#003844", "borderRadius": "18px", "padding": "24px 12px", "boxShadow": "0 4px 24px 0 rgba(57,255,20,0.10)"})
                ], label="Anomaly Log", tab_id="tab-log")
            ], id="main-tabs", active_tab="tab-rt")
        ], width=12)
    ], className="mb-5"),

    # Live Simulated Arduino Data
    dbc.Row([
        dbc.Col([
            html.H1("Live Simulated Arduino Data"),
            html.Div(id='sim-temp', style={'fontSize': 24, 'margin': '10px'}),
            html.Div(id='sim-humidity', style={'fontSize': 24, 'margin': '10px'}),
            html.Div(id='sim-voltage', style={'fontSize': 24, 'margin': '10px'}),
            html.Div(id='sim-current', style={'fontSize': 24, 'margin': '10px'}),
            dcc.Interval(id='sim-interval', interval=2000, n_intervals=0)
        ])
    ], className="mb-5"),
], fluid=True, style={"backgroundColor": "#1a1a1a", "minHeight": "100vh"})

# Callbacks for Arduino Integration
@app.callback(
    Output('arduino-status-display', 'children'),
    Output('arduino-status-display', 'className'),
    Input('interval-component', 'n_intervals')
)
def update_arduino_status(n):
    """Update Arduino connection status"""
    status = get_arduino_status()
    
    if status:
        data_points = status.get('data_points_received', 0)
        models_loaded = status.get('models_loaded', 0)
        last_update = status.get('last_update', 'Unknown')
        
        status_text = f"Arduino Connected - Data Points: {data_points} | Models: {models_loaded} | Last Update: {last_update}"
        status_class = "arduino-connected arduino-status text-center"
    else:
        status_text = "Arduino Disconnected - Check if arduino_data_receiver.py is running"
        status_class = "arduino-disconnected arduino-status text-center"
    
    return status_text, status_class

# Callback for real-time monitoring with synthetic data
@app.callback(
    [Output('live-temperature', 'figure'),
     Output('live-current', 'figure'),
     Output('live-humidity', 'figure'),
     Output('live-vibration', 'figure'),
     Output('live-pressure', 'figure'),
     Output('live-viscosity', 'figure'),
     Output('live-power', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_real_time_monitoring(n):
    """Update real-time monitoring graphs with synthetic data"""
    # Get the last 100 data points for real-time display
    df = combined_df.tail(100)
    
    # Dark theme colors
    dark_bg = '#1a1a1a'
    dark_paper = '#2d3748'
    text_color = '#ffffff'
    grid_color = '#4a5568'
    
    # Temperature real-time
    temp_fig = go.Figure()
    temp_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['temperature'],
        mode='lines',
        name='Temperature',
        line=dict(color='#63b3ed', width=2)
    ))
    temp_fig.update_layout(
        title='Real-time Temperature',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Current real-time
    current_fig = go.Figure()
    current_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['current'],
        mode='lines',
        name='Current',
        line=dict(color='#f6e05e', width=2)
    ))
    current_fig.update_layout(
        title='Real-time Current',
        xaxis_title='Time',
        yaxis_title='Current (A)',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Humidity real-time
    humidity_fig = go.Figure()
    humidity_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['humidity'],
        mode='lines',
        name='Humidity',
        line=dict(color='#63b3ed', width=2)
    ))
    humidity_fig.update_layout(
        title='Real-time Humidity',
        xaxis_title='Time',
        yaxis_title='Humidity (%)',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Vibration real-time
    vibration_fig = go.Figure()
    vibration_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['vibration'],
        mode='lines',
        name='Vibration',
        line=dict(color='#68d391', width=2)
    ))
    vibration_fig.update_layout(
        title='Real-time Vibration',
        xaxis_title='Time',
        yaxis_title='Vibration',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Pressure real-time
    pressure_fig = go.Figure()
    pressure_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['pressure'],
        mode='lines',
        name='Pressure',
        line=dict(color='#fc8181', width=2)
    ))
    pressure_fig.update_layout(
        title='Real-time Pressure',
        xaxis_title='Time',
        yaxis_title='Pressure',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Viscosity real-time
    viscosity_fig = go.Figure()
    viscosity_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['viscosity'],
        mode='lines',
        name='Viscosity',
        line=dict(color='#9f7aea', width=2)
    ))
    viscosity_fig.update_layout(
        title='Real-time Viscosity',
        xaxis_title='Time',
        yaxis_title='Viscosity',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Power real-time
    power_fig = go.Figure()
    power_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['power'],
        mode='lines',
        name='Power',
        line=dict(color='#f6e05e', width=2)
    ))
    power_fig.update_layout(
        title='Real-time Power Consumption',
        xaxis_title='Time',
        yaxis_title='Power (W)',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    return temp_fig, current_fig, humidity_fig, vibration_fig, pressure_fig, viscosity_fig, power_fig

@app.callback(
    Output('ml-anomaly-alert', 'children'),
    Output('ml-anomaly-alert', 'className'),
    Input('interval-component', 'n_intervals')
)
def update_ml_anomaly_alerts(n):
    stats = get_anomaly_stats()
    total_anomalies = stats["total_anomalies"]
    if total_anomalies == 0:
        return (
            html.Span([
                html.Span("✔️", className="icon"),
                "No ML anomalies detected"
            ]),
            "anomaly-alert mx-auto"
        )
    else:
        # Get current anomalies for details
        data = read_simulated_arduino_data()
        current_anomalies = data.get('ml_anomalies', []) if data else []
        # Format concise anomaly types
        anomaly_types = []
        for anomaly in current_anomalies:
            if "TEMPERATURE" in anomaly:
                anomaly_types.append("🔥 Temp")
            if "CURRENT" in anomaly:
                anomaly_types.append("⚡ Curr")
            if "HUMIDITY" in anomaly:
                anomaly_types.append("💧 Hum")
            if "POWER" in anomaly:
                anomaly_types.append("⚡ Power")
        details = " | ".join(anomaly_types)
        return (
            html.Span([
                html.Span("⚠️", className="icon"),
                f"{total_anomalies} ML Anomaly{'ies' if total_anomalies != 1 else ''} detected",
                html.Span(f"  {details}", className="details") if details else None
            ]),
            "anomaly-alert mx-auto"
        )

@app.callback(
    [Output('arduino-temp-value', 'children'),
     Output('arduino-humidity-value', 'children'),
     Output('arduino-current-value', 'children'),
     Output('arduino-temp-anomaly', 'children'),
     Output('arduino-humidity-anomaly', 'children'),
     Output('arduino-current-anomaly', 'children'),
     Output('arduino-power-value', 'children'),
     Output('arduino-power-anomaly', 'children'),
     Output('sim-temp-graph', 'figure'),
     Output('sim-humidity-graph', 'figure'),
     Output('sim-voltage-graph', 'figure'),
     Output('sim-current-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_simulated_arduino_live(n):
    """Update simulated Arduino live data with improved error handling and memory management"""
    try:
        # Periodic cleanup
        cleanup_data_history()
        
        data = read_simulated_arduino_data()
        now = datetime.now()
        
        # Initialize anomaly indicators
        temp_anomaly = ""
        humidity_anomaly = ""
        current_anomaly = ""
        power_anomaly = ""
        
        if data is not None:
            # Extract and validate values
            temperature = data.get('temperature', 0)
            humidity = data.get('humidity', 0)
            voltage = data.get('voltage', 0)
            current = data.get('current', 0)
            
            # Validate data types
            if not all(isinstance(x, (int, float)) for x in [temperature, humidity, voltage, current]):
                print("⚠️  Invalid data types in Arduino data")
                return get_default_returns()
            
            power = current * voltage
            
            # Check for ML anomalies if available
            ml_anomalies = data.get('ml_anomalies', [])
            
            # Set anomaly indicators based on ML detection
            for anomaly in ml_anomalies:
                if "TEMPERATURE" in anomaly:
                    temp_anomaly = "🚨 ML ANOMALY DETECTED"
                if "HUMIDITY" in anomaly:
                    humidity_anomaly = "🚨 ML ANOMALY DETECTED"
                if "CURRENT" in anomaly:
                    current_anomaly = "🚨 ML ANOMALY DETECTED"
                if "POWER" in anomaly:
                    power_anomaly = "🚨 ML ANOMALY DETECTED"
            
            # Update history for graphs with memory management
            try:
                sim_data_history['temperature'].append(float(temperature))
                sim_data_history['humidity'].append(float(humidity))
                sim_data_history['voltage'].append(float(voltage))
                sim_data_history['current'].append(float(current))
                sim_data_history['time'].append(now)
            except Exception as e:
                print(f"⚠️  Error updating data history: {e}")
                # Clear history if there's an issue
                for key in sim_data_history:
                    sim_data_history[key].clear()
            
            # Prepare figures with error handling
            try:
                def make_fig(y, ylabel):
                    return {
                        'data': [go.Scatter(
                            x=list(sim_data_history['time']), 
                            y=list(sim_data_history[y]), 
                            mode='lines+markers', 
                            line={'color': '#39ff14'}
                        )],
                        'layout': go.Layout(
                            margin={'l': 40, 'r': 10, 't': 20, 'b': 40},
                            xaxis={'title': 'Time', 'tickformat': '%H:%M:%S'},
                            yaxis={'title': ylabel},
                            template='plotly_dark',
                            height=220
                        )
                    }
                
                temp_fig = make_fig('temperature', 'Temperature (°C)')
                humidity_fig = make_fig('humidity', 'Humidity (%)')
                voltage_fig = make_fig('voltage', 'Voltage (V)')
                current_fig = make_fig('current', 'Current (A)')
                
            except Exception as e:
                print(f"⚠️  Error creating figures: {e}")
                temp_fig = humidity_fig = voltage_fig = current_fig = get_default_figure()
            
            # Return values with anomaly indicators
            return (
                f"{temperature:.1f}",  # Temperature value
                f"{humidity:.0f}",     # Humidity value
                f"{current:.3f}",      # Current value
                temp_anomaly,          # Temperature anomaly indicator
                humidity_anomaly,      # Humidity anomaly indicator
                current_anomaly,       # Current anomaly indicator
                f"{power:.1f}",        # Power value
                power_anomaly,         # Power anomaly indicator
                temp_fig,              # Temperature graph
                humidity_fig,          # Humidity graph
                voltage_fig,           # Voltage graph
                current_fig            # Current graph
            )
        else:
            print("⚠️  No valid Arduino data available")
            return get_default_returns()
            
    except Exception as e:
        print(f"⚠️  Error in update_simulated_arduino_live: {e}")
        return get_default_returns()

def get_default_returns():
    """Return default values when data is not available"""
    return (
        "N/A", "", "N/A", "", "N/A", "", "N/A", "",
        get_default_figure(),
        get_default_figure(),
        get_default_figure(),
        get_default_figure()
    )

def get_default_figure():
    """Return a default empty figure"""
    return {
        'data': [], 
        'layout': go.Layout(
            template='plotly_dark', 
            height=220,
            xaxis={'title': 'Time'},
            yaxis={'title': 'Value'}
        )
    }

@app.callback(
    Output('arduino-data-table', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_arduino_data_table(n):
    """Update Arduino data table"""
    data = fetch_arduino_data()
    
    if not data:
        return html.Div("No Arduino data available", className="text-center text-muted")
    
    # Get last 10 data points
    recent_data = data[-10:] if len(data) > 10 else data
    
    # Create table
    table_header = [
        html.Thead(html.Tr([
            html.Th("Timestamp", className="text-center"),
            html.Th("Temperature (°C)", className="text-center"),
            html.Th("Humidity (%)", className="text-center"),
            html.Th("Current (A)", className="text-center"),
            html.Th("Vibration (g)", className="text-center"),
            html.Th("Pressure (hPa)", className="text-center"),
            html.Th("Power (W)", className="text-center"),
            html.Th("Anomalies", className="text-center")
        ]))
    ]
    
    table_rows = []
    for row in reversed(recent_data):  # Show newest first
        timestamp = row.get('timestamp', 'N/A')
        if isinstance(timestamp, str) and len(timestamp) > 19:
            timestamp = timestamp[11:19]  # Show only time part
        
        # Check for anomalies
        anomalies = []
        if row.get('is_anomaly_temperature', False):
            anomalies.append("Temp")
        if row.get('is_anomaly_humidity', False):
            anomalies.append("Hum")
        if row.get('is_anomaly_current', False):
            anomalies.append("Cur")
        if row.get('is_anomaly_vibration', False):
            anomalies.append("Vib")
        if row.get('is_anomaly_pressure', False):
            anomalies.append("Pre")
        if row.get('is_anomaly_power', False):
            anomalies.append("Pow")
        
        anomaly_text = ", ".join(anomalies) if anomalies else "None"
        anomaly_color = "text-danger" if anomalies else "text-success"
        
        table_rows.append(html.Tr([
            html.Td(timestamp, className="text-center"),
            html.Td(f"{row.get('temperature', 'N/A'):.1f}" if isinstance(row.get('temperature'), (int, float)) else "N/A", className="text-center"),
            html.Td(f"{row.get('humidity', 'N/A'):.1f}" if isinstance(row.get('humidity'), (int, float)) else "N/A", className="text-center"),
            html.Td(f"{row.get('current', 'N/A'):.2f}" if isinstance(row.get('current'), (int, float)) else "N/A", className="text-center"),
            html.Td(f"{row.get('vibration', 'N/A'):.3f}" if isinstance(row.get('vibration'), (int, float)) else "N/A", className="text-center"),
            html.Td(f"{row.get('pressure', 'N/A'):.1f}" if isinstance(row.get('pressure'), (int, float)) else "N/A", className="text-center"),
            html.Td(f"{row.get('power', 'N/A'):.0f}" if isinstance(row.get('power'), (int, float)) else "N/A", className="text-center"),
            html.Td(anomaly_text, className=f"text-center {anomaly_color}")
        ]))
    
    table_body = [html.Tbody(table_rows)]
    
    return dbc.Table(table_header + table_body, 
                    bordered=True, 
                    hover=True, 
                    responsive=True,
                    className="data-table")

@app.callback(
    Output('anomaly-plot-temp', 'figure'),
    Output('anomaly-plot-current', 'figure'),
    Output('anomaly-plot-humidity', 'figure'),
    Output('anomaly-plot-vibration', 'figure'),
    Output('anomaly-plot-pressure', 'figure'),
    Output('anomaly-plot-viscosity', 'figure'),
    Output('anomaly-plot-power', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_anomaly_detection(n):
    df = combined_df.tail(500)
    
    # Dark theme colors
    dark_bg = '#1a1a1a'
    dark_paper = '#2d3748'
    text_color = '#ffffff'
    grid_color = '#4a5568'
    
    # Temperature anomalies
    temp_fig = go.Figure()
    temp_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['temperature'],
        mode='lines',
        name='Temperature',
        line=dict(color='#63b3ed', width=2)
    ))
    temp_anomalies = df[df['is_anomaly_temp'] == 1]
    temp_fig.add_trace(go.Scatter(
        x=temp_anomalies['timestamp'],
        y=temp_anomalies['temperature'],
        mode='markers',
        marker=dict(color='#fc8181', size=10, symbol='x'),
        name='Anomalies'
    ))
    temp_fig.update_layout(
        title='Temperature Anomaly Detection',
        xaxis_title='Time',
        yaxis_title='Temperature',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Current anomalies
    current_fig = go.Figure()
    current_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['current'],
        mode='lines',
        name='Current',
        line=dict(color='#f6e05e', width=2)
    ))
    current_anomalies = df[df['is_anomaly_current'] == 1]
    current_fig.add_trace(go.Scatter(
        x=current_anomalies['timestamp'],
        y=current_anomalies['current'],
        mode='markers',
        marker=dict(color='#fc8181', size=10, symbol='x'),
        name='Anomalies'
    ))
    current_fig.update_layout(
        title='Current Anomaly Detection',
        xaxis_title='Time',
        yaxis_title='Current',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Humidity anomalies
    humidity_fig = go.Figure()
    humidity_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['humidity'],
        mode='lines',
        name='Humidity',
        line=dict(color='#63b3ed', width=2)
    ))
    humidity_anomalies = df[df['is_anomaly_humidity'] == 1]
    humidity_fig.add_trace(go.Scatter(
        x=humidity_anomalies['timestamp'],
        y=humidity_anomalies['humidity'],
        mode='markers',
        marker=dict(color='#fc8181', size=10, symbol='x'),
        name='Anomalies'
    ))
    humidity_fig.update_layout(
        title='Humidity Anomaly Detection',
        xaxis_title='Time',
        yaxis_title='Humidity',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Vibration anomalies
    vibration_fig = go.Figure()
    vibration_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['vibration'],
        mode='lines',
        name='Vibration',
        line=dict(color='#68d391', width=2)
    ))
    vibration_anomalies = df[df['is_anomaly_vibration'] == 1]
    vibration_fig.add_trace(go.Scatter(
        x=vibration_anomalies['timestamp'],
        y=vibration_anomalies['vibration'],
        mode='markers',
        marker=dict(color='#fc8181', size=10, symbol='x'),
        name='Anomalies'
    ))
    vibration_fig.update_layout(
        title='Vibration Anomaly Detection',
        xaxis_title='Time',
        yaxis_title='Vibration',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Pressure anomalies
    pressure_fig = go.Figure()
    pressure_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['pressure'],
        mode='lines',
        name='Pressure',
        line=dict(color='#fc8181', width=2)
    ))
    pressure_anomalies = df[df['is_anomaly_pressure'] == 1]
    pressure_fig.add_trace(go.Scatter(
        x=pressure_anomalies['timestamp'],
        y=pressure_anomalies['pressure'],
        mode='markers',
        marker=dict(color='#fc8181', size=10, symbol='x'),
        name='Anomalies'
    ))
    pressure_fig.update_layout(
        title='Pressure Anomaly Detection',
        xaxis_title='Time',
        yaxis_title='Pressure',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Viscosity anomalies
    viscosity_fig = go.Figure()
    viscosity_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['viscosity'],
        mode='lines',
        name='Viscosity',
        line=dict(color='#9f7aea', width=2)
    ))
    viscosity_anomalies = df[df['is_anomaly_viscosity'] == 1]
    viscosity_fig.add_trace(go.Scatter(
        x=viscosity_anomalies['timestamp'],
        y=viscosity_anomalies['viscosity'],
        mode='markers',
        marker=dict(color='#fc8181', size=10, symbol='x'),
        name='Anomalies'
    ))
    viscosity_fig.update_layout(
        title='Viscosity Anomaly Detection',
        xaxis_title='Time',
        yaxis_title='Viscosity',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Power anomalies
    power_fig = go.Figure()
    power_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['power'],
        mode='lines',
        name='Power',
        line=dict(color='#f6e05e', width=2)
    ))
    power_anomalies = df[df['is_anomaly_power'] == 1]
    power_fig.add_trace(go.Scatter(
        x=power_anomalies['timestamp'],
        y=power_anomalies['power'],
        mode='markers',
        marker=dict(color='#fc8181', size=10, symbol='x'),
        name='Anomalies'
    ))
    power_fig.update_layout(
        title='Power Anomaly Detection',
        xaxis_title='Time',
        yaxis_title='Power',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    return temp_fig, current_fig, humidity_fig, vibration_fig, pressure_fig, viscosity_fig, power_fig

# Callback for predictive maintenance
@app.callback(
    [Output('maintenance-plot-temp', 'figure'),
     Output('maintenance-plot-current', 'figure'),
     Output('maintenance-plot-humidity', 'figure'),
     Output('maintenance-plot-vibration', 'figure'),
     Output('maintenance-plot-pressure', 'figure'),
     Output('maintenance-plot-viscosity', 'figure'),
     Output('maintenance-plot-power', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_maintenance(n):
    df = combined_df.tail(500)
    
    # Dark theme colors
    dark_bg = '#1a1a1a'
    dark_paper = '#2d3748'
    text_color = '#ffffff'
    grid_color = '#4a5568'
    
    # Temperature maintenance
    temp_fig = go.Figure()
    temp_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['temperature'],
        mode='lines',
        name='Temperature',
        line=dict(color='#63b3ed', width=2)
    ))
    temp_maintenance = df[(df['maintenance_needed'] == 1) & (df['is_anomaly_temp'] == 1)]
    temp_fig.add_trace(go.Scatter(
        x=temp_maintenance['timestamp'],
        y=temp_maintenance['temperature'],
        mode='markers',
        marker=dict(color='#f6e05e', size=12, symbol='diamond'),
        name='Maintenance Needed'
    ))
    temp_fig.update_layout(
        title='Temperature Maintenance',
        xaxis_title='Time',
        yaxis_title='Temperature',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Current maintenance
    current_fig = go.Figure()
    current_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['current'],
        mode='lines',
        name='Current',
        line=dict(color='#f6e05e', width=2)
    ))
    current_maintenance = df[(df['maintenance_needed'] == 1) & (df['is_anomaly_current'] == 1)]
    current_fig.add_trace(go.Scatter(
        x=current_maintenance['timestamp'],
        y=current_maintenance['current'],
        mode='markers',
        marker=dict(color='#f6e05e', size=12, symbol='diamond'),
        name='Maintenance Needed'
    ))
    current_fig.update_layout(
        title='Current Maintenance',
        xaxis_title='Time',
        yaxis_title='Current',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )

    # Humidity maintenance
    humidity_fig = go.Figure()
    humidity_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['humidity'],
        mode='lines',
        name='Humidity',
        line=dict(color='#63b3ed', width=2)
    ))
    humidity_maintenance = df[(df['maintenance_needed'] == 1) & (df['is_anomaly_humidity'] == 1)]
    humidity_fig.add_trace(go.Scatter(
        x=humidity_maintenance['timestamp'],
        y=humidity_maintenance['humidity'],
        mode='markers',
        marker=dict(color='#f6e05e', size=12, symbol='diamond'),
        name='Maintenance Needed'
    ))
    humidity_fig.update_layout(
        title='Humidity Maintenance',
        xaxis_title='Time',
        yaxis_title='Humidity',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Vibration maintenance
    vibration_fig = go.Figure()
    vibration_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['vibration'],
        mode='lines',
        name='Vibration',
        line=dict(color='#68d391', width=2)
    ))
    vibration_maintenance = df[(df['maintenance_needed'] == 1) & (df['is_anomaly_vibration'] == 1)]
    vibration_fig.add_trace(go.Scatter(
        x=vibration_maintenance['timestamp'],
        y=vibration_maintenance['vibration'],
        mode='markers',
        marker=dict(color='#f6e05e', size=12, symbol='diamond'),
        name='Maintenance Needed'
    ))
    vibration_fig.update_layout(
        title='Vibration Maintenance',
        xaxis_title='Time',
        yaxis_title='Vibration',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Pressure maintenance
    pressure_fig = go.Figure()
    pressure_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['pressure'],
        mode='lines',
        name='Pressure',
        line=dict(color='#fc8181', width=2)
    ))
    pressure_maintenance = df[(df['maintenance_needed'] == 1) & (df['is_anomaly_pressure'] == 1)]
    pressure_fig.add_trace(go.Scatter(
        x=pressure_maintenance['timestamp'],
        y=pressure_maintenance['pressure'],
        mode='markers',
        marker=dict(color='#f6e05e', size=12, symbol='diamond'),
        name='Maintenance Needed'
    ))
    pressure_fig.update_layout(
        title='Pressure Maintenance',
        xaxis_title='Time',
        yaxis_title='Pressure',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Viscosity maintenance
    viscosity_fig = go.Figure()
    viscosity_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['viscosity'],
        mode='lines',
        name='Viscosity',
        line=dict(color='#9f7aea', width=2)
    ))
    viscosity_maintenance = df[(df['maintenance_needed'] == 1) & (df['is_anomaly_viscosity'] == 1)]
    viscosity_fig.add_trace(go.Scatter(
        x=viscosity_maintenance['timestamp'],
        y=viscosity_maintenance['viscosity'],
        mode='markers',
        marker=dict(color='#f6e05e', size=12, symbol='diamond'),
        name='Maintenance Needed'
    ))
    viscosity_fig.update_layout(
        title='Viscosity Maintenance',
        xaxis_title='Time',
        yaxis_title='Viscosity',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    # Power maintenance
    power_fig = go.Figure()
    power_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['power'],
        mode='lines',
        name='Power',
        line=dict(color='#f6e05e', width=2)
    ))
    power_maintenance = df[(df['maintenance_needed'] == 1) & (df['is_anomaly_power'] == 1)]
    power_fig.add_trace(go.Scatter(
        x=power_maintenance['timestamp'],
        y=power_maintenance['power'],
        mode='markers',
        marker=dict(color='#f6e05e', size=12, symbol='diamond'),
        name='Maintenance Needed'
    ))
    power_fig.update_layout(
        title='Power Maintenance',
        xaxis_title='Time',
        yaxis_title='Power',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    return temp_fig, current_fig, humidity_fig, vibration_fig, pressure_fig, viscosity_fig, power_fig

# Callback for energy consumption analysis
@app.callback(
    [Output('energy-plot', 'figure'),
     Output('energy-insights', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_energy(n):
    df = combined_df.tail(1440)
    power = df['power']
    energy = power.sum() / 1000  # kWh
    peak_power = power.max()
    avg_power = power.mean()
    
    # Dark theme colors
    dark_bg = '#1a1a1a'
    dark_paper = '#2d3748'
    text_color = '#ffffff'
    grid_color = '#4a5568'
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=power,
        mode='lines',
        name='Power (W)',
        line=dict(color='#f6e05e', width=2)
    ))
    fig.update_layout(
        title='Energy Consumption (Last 24 Hours)',
        xaxis_title='Time',
        yaxis_title='Power (W)',
        plot_bgcolor=dark_paper,
        paper_bgcolor=dark_bg,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
    )
    
    insights = html.Div([
        html.P(f"Total energy (last 24h): {energy:.2f} kWh", 
               style={"color": "#ffffff", "marginBottom": "0.5rem"}),
        html.P(f"Peak power: {peak_power:.2f} W", 
               style={"color": "#ffffff", "marginBottom": "0.5rem"}),
        html.P(f"Average power: {avg_power:.2f} W", 
               style={"color": "#ffffff", "marginBottom": "0.5rem"})
    ])
    
    return fig, insights

@app.callback(
    [Output('sim-temp', 'children'),
     Output('sim-humidity', 'children'),
     Output('sim-voltage', 'children'),
     Output('sim-current', 'children')],
    [Input('sim-interval', 'n_intervals')]
)
def update_simulated_arduino_display(n):
    """Update simulated Arduino display"""
    data = read_simulated_arduino_data()
    
    if data is not None:
        temp = data.get('temperature', 'N/A')
        humidity = data.get('humidity', 'N/A')
        voltage = data.get('voltage', 'N/A')
        current = data.get('current', 'N/A')
        
        return (
            f"Temperature: {temp}°C",
            f"Humidity: {humidity}%",
            f"Voltage: {voltage}V",
            f"Current: {current}A"
        )
    else:
        return "No data", "No data", "No data", "No data"

# Anomaly Log Callbacks
@app.callback(
    Output('anomaly-stats-display', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_anomaly_stats(n):
    """Update anomaly statistics display"""
    stats = get_anomaly_stats()
    
    if stats["total_anomalies"] == 0:
        return html.Div([
            html.P("No anomalies logged yet", className="text-muted"),
            html.P(f"App started: {APP_START_TIME}", className="text-info", style={"fontSize": "0.9rem"}),
            html.P("Start the Arduino simulation to begin logging anomalies", className="text-muted")
        ])
    
    # Create statistics display
    stats_content = []
    
    # App start time
    stats_content.append(
        html.Div([
            html.Strong(f"App Started: {APP_START_TIME}", className="text-info"),
        ], className="mb-2")
    )
    
    # Total anomalies
    stats_content.append(
        html.Div([
            html.Strong(f"Total Anomalies: {stats['total_anomalies']}", className="text-warning"),
            html.Br(),
            html.Small(f"Recent (last 10 min): {stats['recent_count']}", className="text-info")
        ], className="mb-3")
    )
    
    # Anomaly types
    if stats["anomaly_types"]:
        type_text = " | ".join([f"{k}: {v}" for k, v in stats["anomaly_types"].items()])
        stats_content.append(
            html.Div([
                html.Strong("By Type: ", className="text-success"),
                html.Span(type_text, className="text-light")
            ], className="mb-2")
        )
    
    # Severity counts
    if stats["severity_counts"]:
        severity_text = " | ".join([f"{k}: {v}" for k, v in stats["severity_counts"].items()])
        stats_content.append(
            html.Div([
                html.Strong("By Severity: ", className="text-danger"),
                html.Span(severity_text, className="text-light")
            ], className="mb-2")
        )
    
    return html.Div(stats_content)

@app.callback(
    Output('anomaly-log-table', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_anomaly_log_table(n):
    """Update anomaly log table"""
    anomalies = get_anomaly_log_data()
    import datetime
    print(f"[update_anomaly_log_table] Callback fired at {datetime.datetime.now().strftime('%H:%M:%S')}, anomalies count: {len(anomalies)}")
    if not anomalies:
        return html.Div("No anomalies logged yet", className="text-muted")
    
    # Create table header
    table_header = [
        html.Thead(html.Tr([
            html.Th("Timestamp", className="text-center"),
            html.Th("Type", className="text-center"),
            html.Th("Value", className="text-center"),
            html.Th("Severity", className="text-center"),
            html.Th("Sensor Data", className="text-center")
        ]))
    ]
    
    # Create table rows (show most recent first)
    table_rows = []
    for anomaly in reversed(anomalies[-20:]):  # Show last 20 anomalies
        timestamp = anomaly.get('timestamp', 'N/A')
        if len(timestamp) > 19:
            timestamp = timestamp[11:19]  # Show only time part
        
        anomaly_type = anomaly.get('anomaly_type', 'N/A')
        value = anomaly.get('value', 'N/A')
        unit = anomaly.get('unit', '')
        severity = anomaly.get('severity', 'MEDIUM')
        
        # Get sensor data summary
        sensor_data = anomaly.get('sensor_data', {})
        sensor_summary = f"T:{sensor_data.get('temperature', 'N/A')}°C, H:{sensor_data.get('humidity', 'N/A')}%, C:{sensor_data.get('current', 'N/A')}A"
        
        # Color coding for severity
        severity_color = {
            'CRITICAL': 'text-danger',
            'HIGH': 'text-warning',
            'MEDIUM': 'text-info'
        }.get(severity, 'text-light')
        
        table_rows.append(html.Tr([
            html.Td(timestamp, className="text-center"),
            html.Td(anomaly_type, className="text-center"),
            html.Td(f"{value}{unit}", className="text-center"),
            html.Td(severity, className=f"text-center {severity_color}"),
            html.Td(sensor_summary, className="text-center", style={"fontSize": "0.8rem"})
        ]))
    
    table_body = [html.Tbody(table_rows)]
    
    return dbc.Table(table_header + table_body, 
                    bordered=True, 
                    hover=True,
                    responsive=True,
                    className="text-light")

def get_system_health():
    """Get system health status for monitoring"""
    try:
        health = {
            "arduino_file_exists": os.path.exists(SIMULATED_ARDUINO_DATA_PATH),
            "data_history_size": {k: len(v) for k, v in sim_data_history.items()},
            "last_cleanup": last_cleanup_time.strftime("%H:%M:%S"),
            "anomaly_logger_status": "OK" if anomaly_logger else "ERROR"
        }
        return health
    except Exception as e:
        return {"error": str(e)}

def log_system_health():
    """Log system health every 5 minutes"""
    health = get_system_health()
    print(f"📊 System Health: {health}")

if __name__ == '__main__':
    app.run_server(debug=True, host='localhost', port=8050)