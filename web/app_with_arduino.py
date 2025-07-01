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

# Add parent directory to path to import ML models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_models.anomaly_detection import AnomalyDetector
from ml_models.predictive_maintenance import PredictiveMaintenance

class ArduinoIntegratedApp:
    def __init__(self):
        # Arduino data receiver configuration
        self.arduino_server_url = "http://localhost:5001"  # Change to your Arduino receiver server
        self.data_cache = []
        self.max_cache_size = 1000
        
        # Initialize ML models
        self.initialize_models()
        
        # Initialize Dash app with dark theme
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        
        # Setup the layout
        self.setup_layout()
        
        # Setup callbacks
        self.setup_callbacks()
        
        # Start background data fetching
        self.start_background_data_fetching()
    
    def initialize_models(self):
        """Initialize ML models for anomaly detection"""
        try:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            
            # Initialize detectors for each sensor
            self.detectors = {}
            sensors = ['temperature', 'current', 'humidity', 'vibration', 'pressure', 'viscosity', 'power']
            
            for sensor in sensors:
                detector = AnomalyDetector()
                model_path = os.path.join(models_dir, f'{sensor}_anomaly')
                if os.path.exists(model_path):
                    detector.load_models(model_path)
                    self.detectors[sensor] = detector
                    print(f"Loaded {sensor} anomaly detection model")
                else:
                    print(f"Model not found for {sensor}, using default detector")
                    self.detectors[sensor] = detector
            
            # Initialize maintenance model
            self.maintenance_model = PredictiveMaintenance()
            maintenance_path = os.path.join(models_dir, 'maintenance')
            if os.path.exists(maintenance_path):
                self.maintenance_model.load_model(maintenance_path)
                print("Loaded predictive maintenance model")
            
        except Exception as e:
            print(f"Error initializing models: {e}")
    
    def setup_layout(self):
        """Setup the Dash app layout"""
        # Custom CSS for better dark mode styling
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>ENMOS Monitoring System - Arduino Integration</title>
                {%favicon%}
                {%css%}
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
                    .arduino-status {
                        padding: 10px;
                        border-radius: 8px;
                        margin: 10px 0;
                        font-weight: bold;
                    }
                    .arduino-connected {
                        background-color: rgba(57, 255, 20, 0.2);
                        border: 2px solid #39ff14;
                        color: #39ff14;
                    }
                    .arduino-disconnected {
                        background-color: rgba(255, 94, 94, 0.2);
                        border: 2px solid #ff5e5e;
                        color: #ff5e5e;
                    }
                </style>
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
            </head>
            <body>
                <nav class="navbar navbar-expand-lg navbar-dark shadow" style="min-height:90px;">
            <div class="container-fluid">
                <a class="navbar-brand d-flex align-items-center" href="/" style="gap: 16px;">
                    <img src="/assets/enmos_logo.png" alt="ENMOS Logo" style="height: 70px; border-radius: 12px; box-shadow: 0 2px 12px #39ff1499;">
                    <span style="font-size:2.1rem; font-weight:800; letter-spacing:2px; color:#39ff14; text-shadow:0 2px 8px #003844;">ENMOS</span>
                </a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse justify-content-end" id="navbarNav">
                    <ul class="navbar-nav ms-auto align-items-center" style="gap: 8px;">
                        <li class="nav-item">
                            <a class="nav-link fw-bold px-4 py-2" href="/" style="color:#39ff14; border-radius:12px; transition:background 0.2s;">Home</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link fw-bold px-4 py-2" href="/dashboard" style="color:#39ff14; border-radius:12px; transition:background 0.2s;">Dashboard</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>
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
        self.app.layout = dbc.Container([
            # Space below navbar
            html.Div(style={"height": "20px"}),

            # Arduino Connection Status
            dbc.Row([
                dbc.Col([
                    html.Div(id='arduino-status', className='arduino-disconnected arduino-status text-center'),
                ])
            ]),

            # Novice Section
            dbc.Row([
                dbc.Col([
                    html.H2("Real-time Arduino Data", className="text-center mb-3 fw-light", style={"color": "#baff39", "letterSpacing": "2px"}),
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
                                            html.Div(id='temp-value', className="text-muted", style={"fontSize": "1.2rem", "fontWeight": "bold"}),
                                            html.Div(id='temp-anomalies', className="text-danger", style={"fontSize": "0.9rem"})
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
                                            html.Div(id='current-value', className="text-muted", style={"fontSize": "1.2rem", "fontWeight": "bold"}),
                                            html.Div(id='current-anomalies', className="text-danger", style={"fontSize": "0.9rem"})
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
                                            html.I(className="bi bi-droplet", style={"fontSize": "2rem", "color": "#4299e1"}),
                                            html.Div("Humidity", className="fw-semibold mt-2 mb-1", style={"fontSize": "1rem", "color": "#ffffff"}),
                                            html.Div(id='humidity-value', className="text-muted", style={"fontSize": "1.2rem", "fontWeight": "bold"}),
                                            html.Div(id='humidity-anomalies', className="text-danger", style={"fontSize": "0.9rem"})
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
                                            html.I(className="bi bi-vibrate", style={"fontSize": "2rem", "color": "#ed8936"}),
                                            html.Div("Vibration", className="fw-semibold mt-2 mb-1", style={"fontSize": "1rem", "color": "#ffffff"}),
                                            html.Div(id='vibration-value', className="text-muted", style={"fontSize": "1.2rem", "fontWeight": "bold"}),
                                            html.Div(id='vibration-anomalies', className="text-danger", style={"fontSize": "0.9rem"})
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
                                            html.I(className="bi bi-speedometer2", style={"fontSize": "2rem", "color": "#9f7aea"}),
                                            html.Div("Pressure", className="fw-semibold mt-2 mb-1", style={"fontSize": "1rem", "color": "#ffffff"}),
                                            html.Div(id='pressure-value', className="text-muted", style={"fontSize": "1.2rem", "fontWeight": "bold"}),
                                            html.Div(id='pressure-anomalies', className="text-danger", style={"fontSize": "0.9rem"})
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
                                            html.I(className="bi bi-lightning", style={"fontSize": "2rem", "color": "#f56565"}),
                                            html.Div("Power", className="fw-semibold mt-2 mb-1", style={"fontSize": "1rem", "color": "#ffffff"}),
                                            html.Div(id='power-value', className="text-muted", style={"fontSize": "1.2rem", "fontWeight": "bold"}),
                                            html.Div(id='power-anomalies', className="text-danger", style={"fontSize": "0.9rem"})
                                        ], className="text-center")
                                    ])
                                ], className="novice-card shadow-sm border-0 bg-dark mx-1"),
                                style={"minWidth": "170px", "maxWidth": "200px"}
                            ),
                        ], className="justify-content-center")
                    ])
                ])
            ]),

            # Real-time Graphs Section
            dbc.Row([
                dbc.Col([
                    html.H2("Real-time Sensor Data", className="text-center mb-4 fw-light", style={"color": "#baff39", "letterSpacing": "2px"}),
                    dbc.Tabs([
                        dbc.Tab([
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='live-temperature', style={'height': '400px'}), width=6),
                                dbc.Col(dcc.Graph(id='live-current', style={'height': '400px'}), width=6),
                            ]),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='live-humidity', style={'height': '400px'}), width=6),
                                dbc.Col(dcc.Graph(id='live-vibration', style={'height': '400px'}), width=6),
                            ]),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='live-pressure', style={'height': '400px'}), width=6),
                                dbc.Col(dcc.Graph(id='live-power', style={'height': '400px'}), width=6),
                            ])
                        ], label="Live Data", tab_id="live-data"),
                        
                        dbc.Tab([
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='anomaly-plot-temp', style={'height': '400px'}), width=6),
                                dbc.Col(dcc.Graph(id='anomaly-plot-current', style={'height': '400px'}), width=6),
                            ]),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='anomaly-plot-humidity', style={'height': '400px'}), width=6),
                                dbc.Col(dcc.Graph(id='anomaly-plot-vibration', style={'height': '400px'}), width=6),
                            ]),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='anomaly-plot-pressure', style={'height': '400px'}), width=6),
                                dbc.Col(dcc.Graph(id='anomaly-plot-power', style={'height': '400px'}), width=6),
                            ])
                        ], label="Anomaly Detection", tab_id="anomaly-detection"),
                        
                        dbc.Tab([
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='maintenance-plot-temp', style={'height': '400px'}), width=6),
                                dbc.Col(dcc.Graph(id='maintenance-plot-current', style={'height': '400px'}), width=6),
                            ]),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='maintenance-plot-humidity', style={'height': '400px'}), width=6),
                                dbc.Col(dcc.Graph(id='maintenance-plot-vibration', style={'height': '400px'}), width=6),
                            ]),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='maintenance-plot-pressure', style={'height': '400px'}), width=6),
                                dbc.Col(dcc.Graph(id='maintenance-plot-power', style={'height': '400px'}), width=6),
                            ])
                        ], label="Predictive Maintenance", tab_id="maintenance"),
                    ], id="tabs", active_tab="live-data")
                ])
            ], className="mt-4"),

            # Interval component for real-time updates
            dcc.Interval(
                id='interval-component',
                interval=2*1000,  # Update every 2 seconds
                n_intervals=0
            )
        ], fluid=True, className="mt-3")
    
    def setup_callbacks(self):
        """Setup Dash callbacks"""
        
        @self.app.callback(
            Output('arduino-status', 'children'),
            Output('arduino-status', 'className'),
            Input('interval-component', 'n_intervals')
        )
        def update_arduino_status(n):
            try:
                response = requests.get(f"{self.arduino_server_url}/api/arduino/status", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    return f"Arduino Connected - Data Points: {data.get('data_points_received', 0)}", "arduino-connected arduino-status text-center"
                else:
                    return "Arduino Disconnected", "arduino-disconnected arduino-status text-center"
            except:
                return "Arduino Disconnected", "arduino-disconnected arduino-status text-center"
        
        @self.app.callback(
            [Output('temp-value', 'children'),
             Output('current-value', 'children'),
             Output('humidity-value', 'children'),
             Output('vibration-value', 'children'),
             Output('pressure-value', 'children'),
             Output('power-value', 'children'),
             Output('temp-anomalies', 'children'),
             Output('current-anomalies', 'children'),
             Output('humidity-anomalies', 'children'),
             Output('vibration-anomalies', 'children'),
             Output('pressure-anomalies', 'children'),
             Output('power-anomalies', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_real_time_values(n):
            try:
                # Get latest data from cache
                if self.data_cache:
                    latest_data = self.data_cache[-1]
                    
                    # Extract values
                    temp_val = latest_data.get('temperature', 'N/A')
                    current_val = latest_data.get('current', 'N/A')
                    humidity_val = latest_data.get('humidity', 'N/A')
                    vibration_val = latest_data.get('vibration', 'N/A')
                    pressure_val = latest_data.get('pressure', 'N/A')
                    power_val = latest_data.get('power', 'N/A')
                    
                    # Format values with units
                    temp_display = f"{temp_val:.1f}°C" if isinstance(temp_val, (int, float)) else "N/A"
                    current_display = f"{current_val:.2f}A" if isinstance(current_val, (int, float)) else "N/A"
                    humidity_display = f"{humidity_val:.1f}%" if isinstance(humidity_val, (int, float)) else "N/A"
                    vibration_display = f"{vibration_val:.3f}g" if isinstance(vibration_val, (int, float)) else "N/A"
                    pressure_display = f"{pressure_val:.1f}hPa" if isinstance(pressure_val, (int, float)) else "N/A"
                    power_display = f"{power_val:.0f}W" if isinstance(power_val, (int, float)) else "N/A"
                    
                    # Check for anomalies
                    temp_anomaly = "⚠️ ANOMALY" if latest_data.get('is_anomaly_temperature', False) else ""
                    current_anomaly = "⚠️ ANOMALY" if latest_data.get('is_anomaly_current', False) else ""
                    humidity_anomaly = "⚠️ ANOMALY" if latest_data.get('is_anomaly_humidity', False) else ""
                    vibration_anomaly = "⚠️ ANOMALY" if latest_data.get('is_anomaly_vibration', False) else ""
                    pressure_anomaly = "⚠️ ANOMALY" if latest_data.get('is_anomaly_pressure', False) else ""
                    power_anomaly = "⚠️ ANOMALY" if latest_data.get('is_anomaly_power', False) else ""
                    
                    return (temp_display, current_display, humidity_display, vibration_display, 
                           pressure_display, power_display, temp_anomaly, current_anomaly, 
                           humidity_anomaly, vibration_anomaly, pressure_anomaly, power_anomaly)
                else:
                    return ("N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "", "", "", "", "", "")
            except Exception as e:
                print(f"Error updating real-time values: {e}")
                return ("N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "", "", "", "", "", "")
        
        @self.app.callback(
            [Output('live-temperature', 'figure'),
             Output('live-current', 'figure'),
             Output('live-humidity', 'figure'),
             Output('live-vibration', 'figure'),
             Output('live-pressure', 'figure'),
             Output('live-power', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_live_graphs(n):
            try:
                if len(self.data_cache) < 2:
                    # Return empty graphs if not enough data
                    empty_fig = go.Figure()
                    empty_fig.update_layout(
                        title="Waiting for Arduino data...",
                        xaxis_title="Time",
                        yaxis_title="Value",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    return [empty_fig] * 6
                
                # Convert cache to DataFrame
                df = pd.DataFrame(self.data_cache)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Show last 60 data points (2 minutes at 2-second intervals)
                df_recent = df.tail(60)
                
                # Create graphs for each sensor
                graphs = []
                sensors = ['temperature', 'current', 'humidity', 'vibration', 'pressure', 'power']
                colors = ['#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3', '#54a0ff', '#5f27cd']
                
                for sensor, color in zip(sensors, colors):
                    fig = go.Figure()
                    
                    if sensor in df_recent.columns:
                        fig.add_trace(go.Scatter(
                            x=df_recent['timestamp'],
                            y=df_recent[sensor],
                            mode='lines+markers',
                            name=sensor.capitalize(),
                            line=dict(color=color, width=2),
                            marker=dict(size=4)
                        ))
                    
                    fig.update_layout(
                        title=f"Live {sensor.capitalize()} Data",
                        xaxis_title="Time",
                        yaxis_title=sensor.capitalize(),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                    )
                    
                    graphs.append(fig)
                
                return graphs
                
            except Exception as e:
                print(f"Error updating live graphs: {e}")
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="Error loading data",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                return [empty_fig] * 6
        
        @self.app.callback(
            [Output('anomaly-plot-temp', 'figure'),
             Output('anomaly-plot-current', 'figure'),
             Output('anomaly-plot-humidity', 'figure'),
             Output('anomaly-plot-vibration', 'figure'),
             Output('anomaly-plot-pressure', 'figure'),
             Output('anomaly-plot-power', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_anomaly_detection(n):
            try:
                if len(self.data_cache) < 10:
                    empty_fig = go.Figure()
                    empty_fig.update_layout(
                        title="Waiting for data...",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    return [empty_fig] * 6
                
                df = pd.DataFrame(self.data_cache)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                graphs = []
                sensors = ['temperature', 'current', 'humidity', 'vibration', 'pressure', 'power']
                
                for sensor in sensors:
                    fig = go.Figure()
                    
                    if sensor in df.columns:
                        # Normal data
                        normal_data = df[df.get(f'is_anomaly_{sensor}', False) == False]
                        if not normal_data.empty:
                            fig.add_trace(go.Scatter(
                                x=normal_data['timestamp'],
                                y=normal_data[sensor],
                                mode='markers',
                                name='Normal',
                                marker=dict(color='green', size=6)
                            ))
                        
                        # Anomaly data
                        anomaly_data = df[df.get(f'is_anomaly_{sensor}', False) == True]
                        if not anomaly_data.empty:
                            fig.add_trace(go.Scatter(
                                x=anomaly_data['timestamp'],
                                y=anomaly_data[sensor],
                                mode='markers',
                                name='Anomaly',
                                marker=dict(color='red', size=8, symbol='x')
                            ))
                    
                    fig.update_layout(
                        title=f"{sensor.capitalize()} Anomaly Detection",
                        xaxis_title="Time",
                        yaxis_title=sensor.capitalize(),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                    )
                    
                    graphs.append(fig)
                
                return graphs
                
            except Exception as e:
                print(f"Error updating anomaly detection: {e}")
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="Error loading data",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                return [empty_fig] * 6
    
    def start_background_data_fetching(self):
        """Start background thread to fetch data from Arduino receiver"""
        def fetch_data():
            while True:
                try:
                    response = requests.get(f"{self.arduino_server_url}/api/arduino/data", timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if data:
                            # Add new data to cache
                            for item in data:
                                self.data_cache.append(item)
                            
                            # Keep cache size manageable
                            if len(self.data_cache) > self.max_cache_size:
                                self.data_cache = self.data_cache[-self.max_cache_size:]
                    
                except Exception as e:
                    print(f"Error fetching Arduino data: {e}")
                
                time.sleep(2)  # Fetch every 2 seconds
        
        # Start background thread
        data_thread = threading.Thread(target=fetch_data, daemon=True)
        data_thread.start()
    
    def run(self, host='0.0.0.0', port=8050, debug=False):
        """Run the Dash app"""
        print(f"Starting ENMOS Arduino Integration Dashboard on {host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

if __name__ == "__main__":
    app = ArduinoIntegratedApp()
    app.run(debug=True) 