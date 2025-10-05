#!/usr/bin/env python3
"""
📊 Standalone Dashboard for Simplified Zabbix Integration

A self-contained dashboard that doesn't depend on parent directory modules.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import json
import logging
from collections import defaultdict, deque

class StandaloneDashboard:
    """Standalone dashboard for real-time monitoring"""
    
    def __init__(self, port=8051, debug=False):
        self.port = port
        self.debug = debug
        self.app = dash.Dash(__name__)
        self.server_thread = None
        self.is_running = False
        
        # Data storage
        self.timestamps = deque(maxlen=1000)
        self.predictions = defaultdict(lambda: deque(maxlen=1000))
        self.actuals = defaultdict(lambda: deque(maxlen=1000))
        self.anomalies = deque(maxlen=1000)
        self.system_status = {
            'last_update': None,
            'cycle_count': 0,
            'errors': 0,
            'connection_status': 'Disconnected'
        }
        
        # Configure layout
        self._setup_layout()
        self._setup_callbacks()
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🏭 Industrial Network Anomaly Detection", 
                       style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.H3("Real-time Zabbix Integration Dashboard", 
                       style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], style={'padding': '20px'}),
            
            # Status Cards
            html.Div([
                html.Div([
                    html.H4("System Status", style={'color': '#2c3e50'}),
                    html.Div(id='system-status', style={'fontSize': '14px'})
                ], className='status-card', style={
                    'width': '24%', 'display': 'inline-block', 'margin': '0.5%',
                    'border': '1px solid #bdc3c7', 'borderRadius': '5px', 'padding': '15px'
                }),
                
                html.Div([
                    html.H4("Connection", style={'color': '#2c3e50'}),
                    html.Div(id='connection-status', style={'fontSize': '14px'})
                ], className='status-card', style={
                    'width': '24%', 'display': 'inline-block', 'margin': '0.5%',
                    'border': '1px solid #bdc3c7', 'borderRadius': '5px', 'padding': '15px'
                }),
                
                html.Div([
                    html.H4("Predictions", style={'color': '#2c3e50'}),
                    html.Div(id='prediction-stats', style={'fontSize': '14px'})
                ], className='status-card', style={
                    'width': '24%', 'display': 'inline-block', 'margin': '0.5%',
                    'border': '1px solid #bdc3c7', 'borderRadius': '5px', 'padding': '15px'
                }),
                
                html.Div([
                    html.H4("Anomalies", style={'color': '#2c3e50'}),
                    html.Div(id='anomaly-stats', style={'fontSize': '14px'})
                ], className='status-card', style={
                    'width': '24%', 'display': 'inline-block', 'margin': '0.5%',
                    'border': '1px solid #bdc3c7', 'borderRadius': '5px', 'padding': '15px'
                })
            ], style={'padding': '10px'}),
            
            # Main Charts
            html.Div([
                # Time Series Plot
                html.Div([
                    html.H4("📈 Real-time Predictions vs Actuals", style={'color': '#2c3e50'}),
                    dcc.Graph(id='timeseries-plot', style={'height': '400px'})
                ], style={'width': '100%', 'padding': '10px'}),
                
                # Metrics Grid
                html.Div([
                    html.Div([
                        html.H4("🔍 Individual Metrics", style={'color': '#2c3e50'}),
                        dcc.Graph(id='metrics-grid', style={'height': '350px'})
                    ], style={'width': '70%', 'display': 'inline-block', 'padding': '10px'}),
                    
                    html.Div([
                        html.H4("⚠️ Anomaly Detection", style={'color': '#2c3e50'}),
                        dcc.Graph(id='anomaly-plot', style={'height': '350px'})
                    ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'})
                ])
            ]),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            )
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('system-status', 'children'),
             Output('connection-status', 'children'),
             Output('prediction-stats', 'children'),
             Output('anomaly-stats', 'children'),
             Output('timeseries-plot', 'figure'),
             Output('metrics-grid', 'figure'),
             Output('anomaly-plot', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            return self._update_all_components()
    
    def _update_all_components(self):
        """Update all dashboard components"""
        try:
            # System status
            system_status = self._get_system_status_display()
            connection_status = self._get_connection_status_display()
            prediction_stats = self._get_prediction_stats_display()
            anomaly_stats = self._get_anomaly_stats_display()
            
            # Charts
            timeseries_fig = self._create_timeseries_plot()
            metrics_fig = self._create_metrics_grid()
            anomaly_fig = self._create_anomaly_plot()
            
            return (system_status, connection_status, prediction_stats, 
                   anomaly_stats, timeseries_fig, metrics_fig, anomaly_fig)
            
        except Exception as e:
            self.logger.error(f"Dashboard update error: {e}")
            return self._get_error_displays()
    
    def _get_system_status_display(self):
        """Get system status display"""
        if self.system_status['last_update']:
            time_diff = datetime.now() - self.system_status['last_update']
            status_color = '#27ae60' if time_diff.seconds < 120 else '#e74c3c'
            return html.Div([
                html.P(f"🔄 Cycles: {self.system_status['cycle_count']}", 
                      style={'margin': '5px', 'color': status_color}),
                html.P(f"❌ Errors: {self.system_status['errors']}", 
                      style={'margin': '5px'}),
                html.P(f"🕒 Last: {self.system_status['last_update'].strftime('%H:%M:%S')}", 
                      style={'margin': '5px'})
            ])
        else:
            return html.P("⏳ Waiting for data...", style={'color': '#95a5a6'})
    
    def _get_connection_status_display(self):
        """Get connection status display"""
        status = self.system_status['connection_status']
        color = '#27ae60' if status == 'Connected' else '#e74c3c'
        return html.Div([
            html.P(f"🌐 {status}", style={'color': color, 'fontWeight': 'bold'}),
            html.P(f"📊 Data points: {len(self.timestamps)}", style={'margin': '5px'}),
            html.P(f"🔗 Variables: {len(self.predictions)}", style={'margin': '5px'})
        ])
    
    def _get_prediction_stats_display(self):
        """Get prediction statistics display"""
        if len(self.timestamps) > 0:
            total_predictions = sum(len(pred) for pred in self.predictions.values())
            return html.Div([
                html.P(f"📈 Total: {total_predictions}", style={'margin': '5px'}),
                html.P(f"🎯 Variables: {len(self.predictions)}", style={'margin': '5px'}),
                html.P(f"⏱️ Rate: {len(self.timestamps)/max(1, len(self.timestamps)/60):.1f}/min", 
                      style={'margin': '5px'})
            ])
        else:
            return html.P("📊 No predictions yet", style={'color': '#95a5a6'})
    
    def _get_anomaly_stats_display(self):
        """Get anomaly statistics display"""
        if len(self.anomalies) > 0:
            recent_anomalies = sum(1 for a in list(self.anomalies)[-20:] if a)
            return html.Div([
                html.P(f"⚠️ Recent: {recent_anomalies}/20", 
                      style={'margin': '5px', 'color': '#e74c3c' if recent_anomalies > 5 else '#27ae60'}),
                html.P(f"📊 Total: {sum(self.anomalies)}", style={'margin': '5px'}),
                html.P(f"📈 Rate: {(sum(self.anomalies)/max(1, len(self.anomalies))*100):.1f}%", 
                      style={'margin': '5px'})
            ])
        else:
            return html.P("✅ No anomalies detected", style={'color': '#27ae60'})
    
    def _create_timeseries_plot(self):
        """Create the main time series plot"""
        fig = go.Figure()
        
        if len(self.timestamps) == 0:
            fig.add_annotation(
                text="⏳ Waiting for real-time data...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#95a5a6')
            )
        else:
            timestamps = list(self.timestamps)
            
            # Plot each variable
            for var_name, predictions in self.predictions.items():
                if len(predictions) > 0:
                    pred_values = list(predictions)
                    # Ensure same length as timestamps
                    if len(pred_values) == len(timestamps):
                        fig.add_trace(go.Scatter(
                            x=timestamps, y=pred_values,
                            mode='lines+markers',
                            name=f'{var_name} (Predicted)',
                            line=dict(width=2)
                        ))
                    
                    # Add actuals if available
                    if var_name in self.actuals and len(self.actuals[var_name]) > 0:
                        actual_values = list(self.actuals[var_name])
                        if len(actual_values) == len(timestamps):
                            fig.add_trace(go.Scatter(
                                x=timestamps, y=actual_values,
                                mode='lines',
                                name=f'{var_name} (Actual)',
                                line=dict(dash='dash', width=1),
                                opacity=0.7
                            ))
        
        fig.update_layout(
            title="Real-time Predictions and Actuals",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def _create_metrics_grid(self):
        """Create metrics grid subplot"""
        from plotly.subplots import make_subplots
        
        n_vars = len(self.predictions)
        if n_vars == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="📊 No metrics data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create subplots
        rows = max(1, (n_vars + 1) // 2)
        fig = make_subplots(
            rows=rows, cols=2,
            subplot_titles=list(self.predictions.keys())[:4],  # Limit to first 4
            vertical_spacing=0.1
        )
        
        timestamps = list(self.timestamps)
        
        for i, (var_name, predictions) in enumerate(list(self.predictions.items())[:4]):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            if len(predictions) > 0 and len(predictions) == len(timestamps):
                pred_values = list(predictions)
                fig.add_trace(
                    go.Scatter(x=timestamps, y=pred_values,
                             mode='lines', name=var_name,
                             showlegend=False),
                    row=row, col=col
                )
        
        fig.update_layout(height=350, margin=dict(l=40, r=40, t=40, b=40))
        return fig
    
    def _create_anomaly_plot(self):
        """Create anomaly detection plot"""
        fig = go.Figure()
        
        if len(self.anomalies) == 0:
            fig.add_annotation(
                text="✅ No anomalies detected",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color='#27ae60')
            )
        else:
            timestamps = list(self.timestamps)
            anomaly_values = list(self.anomalies)
            
            if len(anomaly_values) == len(timestamps):
                # Create anomaly indicator
                colors = ['red' if a else 'green' for a in anomaly_values]
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=anomaly_values,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=colors,
                        opacity=0.7
                    ),
                    name='Anomalies'
                ))
        
        fig.update_layout(
            title="Anomaly Detection Status",
            xaxis_title="Time",
            yaxis_title="Anomaly",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def _get_error_displays(self):
        """Get error displays when dashboard update fails"""
        error_msg = html.P("❌ Dashboard update error", style={'color': '#e74c3c'})
        empty_fig = go.Figure()
        return (error_msg, error_msg, error_msg, error_msg, empty_fig, empty_fig, empty_fig)
    
    def update_data(self, timestamp, predictions_dict, actuals_dict=None, is_anomaly=False):
        """Update dashboard data"""
        try:
            self.timestamps.append(timestamp)
            
            # Update predictions
            for var_name, value in predictions_dict.items():
                self.predictions[var_name].append(value)
            
            # Update actuals if provided
            if actuals_dict:
                for var_name, value in actuals_dict.items():
                    self.actuals[var_name].append(value)
            
            # Update anomaly
            self.anomalies.append(is_anomaly)
            
            # Update system status
            self.system_status['last_update'] = datetime.now()
            self.system_status['cycle_count'] += 1
            self.system_status['connection_status'] = 'Connected'
            
        except Exception as e:
            self.logger.error(f"Error updating dashboard data: {e}")
            self.system_status['errors'] += 1
    
    def set_connection_status(self, status):
        """Set connection status"""
        self.system_status['connection_status'] = status
    
    def start_server(self):
        """Start the dashboard server"""
        def run_server():
            try:
                # Use the newer app.run method for newer Dash versions
                self.app.run(
                    debug=self.debug,
                    host='0.0.0.0',
                    port=self.port,
                    use_reloader=False,  # Important: disable reloader in thread
                    dev_tools_hot_reload=False
                )
            except AttributeError:
                # Fallback to run_server for older Dash versions
                self.app.run_server(
                    debug=self.debug,
                    host='0.0.0.0',
                    port=self.port,
                    use_reloader=False,
                    dev_tools_hot_reload=False
                )
            except Exception as e:
                self.logger.error(f"Dashboard server error: {e}")
        
        if not self.is_running:
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            self.is_running = True
            time.sleep(2)  # Give server time to start
            print(f"✅ Dashboard started at http://localhost:{self.port}")
    
    def stop_server(self):
        """Stop the dashboard server"""
        self.is_running = False
        if self.server_thread:
            # Note: Dash doesn't have a clean shutdown method
            # Server will stop when main thread exits
            print("🛑 Dashboard server stopping...")

# Test the dashboard
if __name__ == "__main__":
    import random
    
    dashboard = StandaloneDashboard(port=8051, debug=True)
    dashboard.start_server()
    
    # Simulate some data
    try:
        for i in range(100):
            timestamp = datetime.now() - timedelta(minutes=100-i)
            predictions = {
                'CPU_Usage': random.uniform(20, 80),
                'Memory_Usage': random.uniform(40, 90),
                'Network_Bits': random.uniform(1000, 5000)
            }
            actuals = {k: v + random.uniform(-5, 5) for k, v in predictions.items()}
            is_anomaly = random.random() < 0.1
            
            dashboard.update_data(timestamp, predictions, actuals, is_anomaly)
            time.sleep(0.1)
        
        print("📊 Test data loaded. Dashboard running at http://localhost:8051")
        print("Press Ctrl+C to stop...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard...")
        dashboard.stop_server()