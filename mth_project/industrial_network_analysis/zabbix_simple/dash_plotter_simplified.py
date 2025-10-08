#!/usr/bin/env python3
"""
Simplified Dash Real-Time Plotter for Industrial Network Forecasting
Fixed timestep alignment and simplified UI while keeping all functionality
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import re


class SimplifiedDashPlotter:
    """
    Simplified real-time dashboard with correct time alignment
    Fixes the 6-timestep shift issue and maintains clean UI
    """
    
    def __init__(self, max_points=120, update_interval=60000):
        """
        Initialize simplified dashboard
        
        Args:
            max_points: Maximum points to display (default 2 hours at 1min intervals)
            update_interval: Update interval in milliseconds
        """
        self.app = dash.Dash(__name__)
        self.max_points = max_points
        self.update_interval = update_interval
        
        # Core data storage
        self.timestamps = deque(maxlen=max_points)
        self.actual_values = {}           # Historical actual values
        self.predictions_t1 = {}          # One-step ahead predictions (t+1)
        self.future_predictions = {}      # Multi-step predictions (t+1 to t+6)
        self.variable_names = []
        
        # Progress tracking
        self.current_step = 0
        self.total_steps = 0
        
        # Classification and alerts
        self.classification_results = deque(maxlen=20)
        self.port_status = {}
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'last_update': None,
            'avg_error': 0.0
        }
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup clean and organized layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🏭 Network Traffic Forecasting Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px'})
            ]),
            
            # Status bar
            html.Div(id='status-bar', style={
                'textAlign': 'center', 
                'backgroundColor': '#ecf0f1', 
                'padding': '10px', 
                'borderRadius': '5px',
                'margin': '10px 20px'
            }),
            
            # Classification alerts (if any)
            html.Div(id='alerts-section', style={'margin': '20px'}),
            
            # Main graphs container
            html.Div(id='graphs-container', style={'margin': '20px'}),
            
            # Port status (if available)
            html.Div(id='port-status', style={
                'margin': '20px',
                'padding': '15px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '5px'
            }),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            )
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            [Output('status-bar', 'children'),
             Output('alerts-section', 'children'),
             Output('graphs-container', 'children'),
             Output('port-status', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            return (
                self._get_status_bar(),
                self._get_alerts_section(),
                self._get_main_graphs(),
                self._get_port_status()
            )
    
    def _get_status_bar(self):
        """Create status information bar"""
        progress_pct = (self.current_step / max(self.total_steps, 1)) * 100 if self.total_steps > 0 else 0
        
        return html.Div([
            html.Span(f"📊 Progress: {self.current_step}/{self.total_steps} ({progress_pct:.1f}%) | ", 
                     style={'marginRight': '20px'}),
            html.Span(f"🎯 Total Predictions: {self.stats['total_predictions']} | ", 
                     style={'marginRight': '20px'}),
            html.Span(f"📈 Variables: {len(self.variable_names)} | ", 
                     style={'marginRight': '20px'}),
            html.Span(f"🕒 Last Update: {self.stats['last_update'] or 'Never'}")
        ], style={'fontSize': '14px', 'color': '#34495e'})
    
    def _get_alerts_section(self):
        """Show classification alerts if any"""
        if not self.classification_results:
            return html.Div()
        
        # Show only recent critical alerts
        recent_alerts = list(self.classification_results)[-3:]
        alert_items = []
        
        for result in recent_alerts:
            timestamp = result.get('timestamp', 'N/A')
            classification = result.get('classification', 'Unknown')
            confidence = result.get('confidence', 0)
            
            # Determine alert level
            is_critical = 'storm' in str(classification).lower() or 'anomaly' in str(classification).lower()
            
            if is_critical:
                color = '#e74c3c'
                icon = '🚨'
                bg_color = '#ffe6e6'
            else:
                color = '#27ae60'
                icon = '✅'
                bg_color = '#e8f5e8'
            
            alert_items.append(
                html.Div([
                    html.Span(f"{icon} {timestamp}: {classification} ", 
                             style={'fontWeight': 'bold', 'color': color}),
                    html.Span(f"(Confidence: {confidence:.2f})", 
                             style={'color': '#666', 'fontSize': '12px'})
                ], style={
                    'padding': '8px',
                    'margin': '5px 0',
                    'backgroundColor': bg_color,
                    'borderRadius': '5px',
                    'border': f'1px solid {color}'
                })
            )
        
        if alert_items:
            return html.Div([
                html.H4("🚨 System Alerts", style={'color': '#e74c3c', 'margin': '10px 0'}),
                html.Div(alert_items)
            ])
        
        return html.Div()
    
    def _get_main_graphs(self):
        """Create main forecasting graphs"""
        if not self.variable_names or not self.timestamps:
            return html.Div("Waiting for data...", 
                          style={'textAlign': 'center', 'padding': '50px', 'color': '#666'})
        
        # Categorize variables for better organization
        numeric_vars, port_vars = self._categorize_variables()
        
        sections = []
        
        # System metrics section
        if numeric_vars:
            sections.append(html.Div([
                html.H3("📊 System Metrics", style={'color': '#3498db', 'margin': '20px 0'}),
                self._create_variable_graphs(numeric_vars, rows=2, cols=2)
            ]))
        
        # Port traffic section
        if port_vars:
            sections.append(html.Div([
                html.H3("🌐 Port Traffic", style={'color': '#9b59b6', 'margin': '20px 0'}),
                self._create_variable_graphs(port_vars, rows=2, cols=2)
            ]))
        
        return html.Div(sections)
    
    def _create_variable_graphs(self, variables, rows=2, cols=2):
        """Create subplot graphs for variables with CORRECT time alignment"""
        if not variables:
            return html.Div("No data available")
        
        # Calculate subplot layout
        num_vars = len(variables)
        if num_vars <= 4:
            rows, cols = 2, 2
        elif num_vars <= 6:
            rows, cols = 2, 3
        else:
            rows = (num_vars + cols - 1) // cols
        
        # Create subplots
        fig = sp.make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=variables,
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # Get current time reference
        current_time = list(self.timestamps)[-1] if self.timestamps else datetime.now()
        
        for i, var in enumerate(variables):
            row = i // cols + 1
            col = i % cols + 1
            
            # === FIX: CORRECT TIME ALIGNMENT ===
            
            # 1. Historical actual values (BLUE) - up to current time
            if var in self.actual_values and len(self.actual_values[var]) > 0:
                historical_data = list(self.actual_values[var])
                historical_times = list(self.timestamps)
                
                # Ensure same length
                min_len = min(len(historical_data), len(historical_times))
                if min_len > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=historical_times[-min_len:],
                            y=historical_data[-min_len:],
                            mode='lines+markers',
                            name='Actual',
                            line=dict(color='#3498db', width=2),
                            marker=dict(size=3),
                            showlegend=(i == 0)
                        ),
                        row=row, col=col
                    )
            
            # 2. One-step predictions (RED) - t+1 prediction aligned with t+1 timestamp
            if var in self.predictions_t1 and len(self.predictions_t1[var]) > 0:
                pred_data = list(self.predictions_t1[var])
                
                # Create t+1 timestamps (shifted by 1 minute into future)
                t1_times = []
                historical_times = list(self.timestamps)
                for i_t, base_time in enumerate(historical_times):
                    if i_t < len(pred_data):
                        t1_times.append(base_time + timedelta(minutes=1))
                
                # Ensure same length
                min_len = min(len(pred_data), len(t1_times))
                if min_len > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=t1_times[-min_len:],
                            y=pred_data[-min_len:],
                            mode='lines+markers',
                            name='Predictions (t+1)',
                            line=dict(color='#e74c3c', width=2, dash='dash'),
                            marker=dict(size=3),
                            showlegend=(i == 0)
                        ),
                        row=row, col=col
                    )
            
            # 3. Future predictions (GREEN) - t+1 through t+6 from current time
            if var in self.future_predictions and len(self.future_predictions[var]) > 0:
                latest_future = self.future_predictions[var][-1] if self.future_predictions[var] else []
                
                if isinstance(latest_future, list) and len(latest_future) > 0:
                    # Create future timestamps starting from current_time + 1min
                    future_times = []
                    future_values = []
                    
                    for step in range(min(6, len(latest_future))):
                        future_time = current_time + timedelta(minutes=step + 1)
                        future_times.append(future_time)
                        future_values.append(latest_future[step])
                    
                    if len(future_times) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=future_times,
                                y=future_values,
                                mode='lines+markers',
                                name='Future (t+1→t+6)',
                                line=dict(color='#27ae60', width=2, dash='dot'),
                                marker=dict(size=4),
                                showlegend=(i == 0)
                            ),
                            row=row, col=col
                        )
        
        # Update layout
        fig.update_layout(
            height=300 * rows,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=80, b=40, l=50, r=50)
        )
        
        # Format x-axis for time
        fig.update_xaxes(tickformat='%H:%M')
        
        return dcc.Graph(figure=fig, style={'height': f'{300 * rows}px'})
    
    def _categorize_variables(self):
        """Categorize variables into system metrics and port traffic"""
        numeric_vars = []
        port_vars = []
        
        for var in self.variable_names:
            var_lower = var.lower()
            
            # Port-related variables
            if any(keyword in var_lower for keyword in ['bits', 'port', 'interface', 'traffic']):
                port_vars.append(var)
            else:
                # System metrics (CPU, memory, etc.)
                numeric_vars.append(var)
        
        return numeric_vars, port_vars
    
    def _get_port_status(self):
        """Display port status information"""
        if not self.port_status:
            return html.Div()
        
        status_items = []
        for port_id, status_info in self.port_status.items():
            status = status_info.get('status', 'Unknown')
            last_update = status_info.get('last_update', 'N/A')
            
            # Status color coding
            if status == 'Active' or status == 'UP':
                color = '#27ae60'
                icon = '🟢'
            elif status == 'DOWN' or 'down' in status.lower():
                color = '#e74c3c'
                icon = '🔴'
            else:
                color = '#f39c12'
                icon = '🟡'
            
            status_items.append(
                html.Span([
                    html.Span(f"{icon} {port_id}: {status}", 
                             style={'color': color, 'fontWeight': 'bold'}),
                    html.Span(f" ({last_update})", 
                             style={'color': '#666', 'fontSize': '11px'})
                ], style={'margin': '5px 15px', 'display': 'inline-block'})
            )
        
        if status_items:
            return html.Div([
                html.H4("🔌 Port Status", style={'color': '#34495e', 'margin': '10px 0'}),
                html.Div(status_items)
            ])
        
        return html.Div()
    
    def add_data_point(self, predictions, actuals, current_step, current_datetime, 
                      variable_names, saved_prediction=None, future_prediction=None, 
                      port_statuses=None, classification_result=None):
        """
        Add new data point with CORRECT time alignment
        
        Args:
            predictions: List of prediction arrays for t+1 through t+6
            actuals: List of actual value arrays (current time)
            current_step: Current step number
            current_datetime: Current timestamp
            variable_names: List of variable names
            saved_prediction: Single-step prediction for t+1
            future_prediction: Multi-step predictions
            port_statuses: Port status information
            classification_result: Classification result
        """
        try:
            # Initialize variables if first time
            if not self.variable_names:
                self.variable_names = variable_names
                for var in variable_names:
                    self.actual_values[var] = deque(maxlen=self.max_points)
                    self.predictions_t1[var] = deque(maxlen=self.max_points)
                    self.future_predictions[var] = deque(maxlen=20)  # Keep fewer future predictions
            
            # Validate inputs
            if not predictions or not actuals or not variable_names:
                return
            
            # Add timestamp
            self.timestamps.append(current_datetime)
            
            # Add actual values (current time t)
            if len(actuals) > 0 and len(actuals[0]) == len(variable_names):
                for i, var in enumerate(variable_names):
                    self.actual_values[var].append(actuals[0][i])
            
            # Add t+1 prediction
            t1_prediction = saved_prediction if saved_prediction else (predictions[0] if predictions else None)
            if t1_prediction and len(t1_prediction) == len(variable_names):
                for i, var in enumerate(variable_names):
                    self.predictions_t1[var].append(t1_prediction[i])
            
            # Add future predictions (full horizon t+1 to t+6)
            if predictions and len(predictions) > 0:
                for i, var in enumerate(variable_names):
                    # Extract all future predictions for this variable
                    var_future_predictions = []
                    for pred_step in predictions:
                        if i < len(pred_step):
                            var_future_predictions.append(pred_step[i])
                    
                    if var_future_predictions:
                        self.future_predictions[var].append(var_future_predictions)
            
            # Update port status
            if port_statuses:
                current_time_str = current_datetime.strftime("%H:%M:%S")
                for port_name, status_value in port_statuses.items():
                    status_text = "UP" if status_value in [1, 1.0, True] else "DOWN"
                    self.port_status[port_name] = {
                        'status': status_text,
                        'last_update': current_time_str,
                        'raw_value': status_value
                    }
            
            # Add classification result
            if classification_result:
                timestamp_str = current_datetime.strftime("%H:%M:%S")
                if isinstance(classification_result, dict):
                    self.classification_results.append({
                        'timestamp': timestamp_str,
                        'classification': classification_result.get('classification', 'Unknown'),
                        'confidence': classification_result.get('confidence', 0.0)
                    })
                else:
                    self.classification_results.append({
                        'timestamp': timestamp_str,
                        'classification': str(classification_result),
                        'confidence': 1.0
                    })
            
            # Update statistics
            self.current_step = current_step
            self.stats['total_predictions'] += 1
            self.stats['last_update'] = datetime.now().strftime("%H:%M:%S")
            
        except Exception as e:
            print(f"Error in add_data_point: {e}")
    
    def set_total_steps(self, total_steps):
        """Set total steps for progress tracking"""
        self.total_steps = total_steps
    
    def clear_data(self):
        """Clear all stored data"""
        self.timestamps.clear()
        for var in self.variable_names:
            if var in self.actual_values:
                self.actual_values[var].clear()
            if var in self.predictions_t1:
                self.predictions_t1[var].clear()
            if var in self.future_predictions:
                self.future_predictions[var].clear()
        
        self.classification_results.clear()
        self.port_status.clear()
        self.stats['total_predictions'] = 0
        self.current_step = 0
    
    def start_server(self, host='127.0.0.1', port=8050, debug=False):
        """Start the Dash server"""
        def run_server():
            self.app.run(host=host, port=port, debug=debug, use_reloader=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        print(f"✅ Simplified dashboard started at http://{host}:{port}")
        return server_thread
    
    def get_statistics(self):
        """Get dashboard statistics"""
        return {
            'total_predictions': self.stats['total_predictions'],
            'total_variables': len(self.variable_names),
            'current_step': self.current_step,
            'total_steps': self.total_steps
        }


# Backward compatibility alias
DashRealTimePlotter = SimplifiedDashPlotter


def test_simplified_dashboard():
    """Test the simplified dashboard"""
    dashboard = SimplifiedDashPlotter()
    
    # Start server
    dashboard.start_server()
    print("Test dashboard started at http://localhost:8050")
    
    # Simulate data
    variables = ['cpu_usage', 'memory_usage', 'bits_sent_port1', 'bits_recv_port1']
    dashboard.set_total_steps(50)
    
    for step in range(30):
        # Generate test data
        predictions = []
        for pred_step in range(6):  # 6-step horizon
            predictions.append(np.random.randn(len(variables)) * 10 + 50 + pred_step)
        
        actuals = [predictions[0] + np.random.randn(len(variables)) * 2]
        timestamp = datetime.now() + timedelta(minutes=step)
        
        # Add data point
        dashboard.add_data_point(
            predictions=predictions,
            actuals=actuals,
            current_step=step,
            current_datetime=timestamp,
            variable_names=variables,
            saved_prediction=predictions[0]
        )
        
        # Add some classification results
        if step % 7 == 0:
            classification = {
                'classification': 'Normal Operation',
                'confidence': 0.95
            }
            dashboard.add_data_point(
                predictions=predictions,
                actuals=actuals,
                current_step=step,
                current_datetime=timestamp,
                variable_names=variables,
                classification_result=classification
            )
        
        time.sleep(0.5)  # Wait between updates
    
    print("Test completed. Dashboard should show properly aligned data.")
    return dashboard


if __name__ == "__main__":
    test_simplified_dashboard()
    input("Press Enter to stop...")