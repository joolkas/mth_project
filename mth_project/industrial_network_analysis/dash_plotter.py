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


class DashRealTimePlotter:
    def __init__(self, max_points=60, update_interval=1000):
        """
        Simplified real-time plotter for industrial network forecasting data.
        
        Args:
            max_points: Maximum number of points to display in each plot
            update_interval: Update interval in milliseconds
        """
        self.app = dash.Dash(__name__)
        self.max_points = max_points
        self.update_interval = update_interval
        
        # Data storage - simplified to essential streams only
        self.timestamps = deque(maxlen=max_points)
        self.actual_values = {}           # Actual values (historical)
        self.saved_predictions = {}       # Saved predictions with latest t+1
        self.temporal_predictions = {}    # Temporal predictions (t+6 horizon)
        self.variable_names = []
        self.current_step = 0
        self.total_steps = 0
        self.prediction_horizon = 6
        
        # Basic statistics
        self.stats = {
            'total_predictions': 0,
            'last_update': None
        }
        
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup simplified Dash app layout with graphs at bottom"""
        self.app.layout = html.Div([
            # Simple Header
            html.Div([
                html.H1("Network Traffic Forecasting Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px'})
            ]),
            
            # Basic Status Info
            html.Div(id='status-info', 
                    style={'textAlign': 'center', 'fontSize': '14px', 'margin': '10px'}),
            
            # Main visualization graphs at bottom
            html.Div(id='graphs-container', style={'margin': '20px'}),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            )
        ])
    
    def _setup_callbacks(self):
        """Setup simplified Dash callbacks"""
        @self.app.callback(
            [Output('graphs-container', 'children'),
             Output('status-info', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            return (
                self._update_graphs(), 
                self._get_status_info()
            )
    
    def _update_graphs(self):
        """Update all graphs with latest data - simplified visualization only"""
        try:
            if not self.variable_names:
                return html.Div("Waiting for data...", style={'textAlign': 'center', 'padding': '50px'})
            
            # Create two-column layout
            num_vars = len(self.variable_names)
            cols = 2
            rows = (num_vars + cols - 1) // cols
            
            # Simple spacing
            vertical_spacing = 0.01
            horizontal_spacing = 0.03
            
            # Simple subplot titles
            subplot_titles = [var for var in self.variable_names]
            
            fig = sp.make_subplots(
                rows=rows, 
                cols=cols,
                subplot_titles=subplot_titles,
                vertical_spacing=vertical_spacing,
                horizontal_spacing=horizontal_spacing
            )
            
            # Add traces for each variable - three traces only
            for i, var in enumerate(self.variable_names):
                row = i // cols + 1
                col = i % cols + 1
                
                timestamps = list(self.timestamps)
                
                # 1. Actual values trace (blue) - exclude last sample
                if var in self.actual_values and len(self.actual_values[var]) > 0:
                    actual_data = list(self.actual_values[var])
                    data_length = min(len(actual_data), len(timestamps))
                    
                    if data_length > 1:
                        recent_timestamps = timestamps[-data_length:-1]
                        recent_actual_data = actual_data[-data_length:-1]
                        
                        if len(recent_timestamps) > 0 and len(recent_actual_data) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=recent_timestamps,
                                    y=recent_actual_data,
                                    mode='lines+markers',
                                    name='Actual Values',
                                    line=dict(color='blue', width=2),
                                    marker=dict(size=4),
                                    showlegend=(i == 0)
                                ),
                                row=row, col=col
                            )
                
                # 2. Saved predictions (red)
                if var in self.saved_predictions and len(self.saved_predictions[var]) > 0:
                    saved_data = list(self.saved_predictions[var])
                    data_length = min(len(saved_data), len(timestamps))
                    
                    if data_length > 0:
                        recent_timestamps = timestamps[-data_length:]
                        recent_saved_data = saved_data[-data_length:]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=recent_timestamps,
                                y=recent_saved_data,
                                mode='lines+markers',
                                name='Predictions (t+1)',
                                line=dict(color='red', width=2, dash='dash'),
                                marker=dict(size=4),
                                showlegend=(i == 0)
                            ),
                            row=row, col=col
                        )
                
                # 3. Temporal predictions (orange)
                if var in self.temporal_predictions and len(self.temporal_predictions[var]) > 0:
                    temporal_data = list(self.temporal_predictions[var])
                    
                    if timestamps and len(temporal_data) > 0:
                        current_time = timestamps[-1] if timestamps else datetime.now()
                        
                        future_timestamps = []
                        future_predictions = []
                        
                        latest_temporal = temporal_data[-1] if temporal_data else []
                        
                        for step in range(0, min(6, len(latest_temporal))):
                            future_time = current_time + timedelta(minutes=step)
                            future_timestamps.append(future_time)
                            
                            if step < len(latest_temporal):
                                future_predictions.append(latest_temporal[step])
                        
                        if len(future_timestamps) > 0 and len(future_predictions) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=future_timestamps,
                                    y=future_predictions,
                                    mode='lines+markers',
                                    name='Future Predictions (t+1 to t+6)',
                                    line=dict(color='orange', width=2, dash='dot'),
                                    marker=dict(size=3),
                                    showlegend=(i == 0)
                                ),
                                row=row, col=col
                            )
            
            # Simple layout
            total_height = 600 * rows
            
            fig.update_layout(
                height=total_height,
                title_text="Real-Time Network Traffic Forecasting",
                title_x=0.5,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Time", showgrid=True, tickformat='%H:%M:%S')
            fig.update_yaxes(title_text="Value", showgrid=True)
            
            return dcc.Graph(figure=fig, style={'height': f'{total_height}px'})
            
        except Exception as e:
            return html.Div(f"Error updating graphs: {str(e)}", 
                          style={'textAlign': 'center', 'padding': '50px', 'color': 'red'})
    
    def _get_status_info(self):
        """Simple status information display"""
        progress = (self.current_step / max(self.total_steps, 1)) * 100 if self.total_steps > 0 else 0
        
        return html.Div([
            html.Span(f"Progress: {self.current_step}/{self.total_steps} ({progress:.1f}%) | "),
            html.Span(f"Predictions: {self.stats['total_predictions']} | "),
            html.Span(f"Variables: {len(self.variable_names)} | "),
            html.Span(f"Last Update: {self.stats['last_update'] or 'Never'}")
        ])

    def set_total_steps(self, total_steps):
        """Set the total number of steps for progress tracking"""
        self.total_steps = total_steps

    def add_buffer_predictions(self, predictions, actuals, current_step, current_datetime, variable_names, 
                             saved_prediction=None, future_prediction=None, port_statuses=None, classification_result=None):
        """
        Add new prediction data to the dashboard - simplified version
        """
        try:
            # Initialize variable names if first time
            if not self.variable_names:
                self.variable_names = variable_names
                for var in variable_names:
                    self.actual_values[var] = deque(maxlen=self.max_points)
                    self.saved_predictions[var] = deque(maxlen=self.max_points)
                    self.temporal_predictions[var] = deque(maxlen=10)
            
            # Validate inputs
            if not predictions or not actuals:
                return
            
            if len(predictions) == 0 or len(actuals) == 0:
                return
                
            # Use saved prediction if provided, otherwise use first prediction
            pred_t1_step = saved_prediction if saved_prediction is not None else predictions[0]
            actual_step = actuals[0]
            
            # Validate data lengths
            if len(pred_t1_step) != len(variable_names) or len(actual_step) != len(variable_names):
                return
            
            # Add timestamp
            self.timestamps.append(current_datetime)
            
            # Add data for each variable
            for i, var in enumerate(variable_names):
                # 1. Actual values
                self.actual_values[var].append(actual_step[i])
                
                # 2. Saved predictions (t+1)
                self.saved_predictions[var].append(pred_t1_step[i])
                
                # 3. Temporal predictions
                if predictions and len(predictions) > 0:
                    var_predictions = []
                    for pred_step in predictions:
                        if i < len(pred_step):
                            var_predictions.append(pred_step[i])
                    
                    if var_predictions:
                        self.temporal_predictions[var].append(var_predictions)
            
            # Update statistics
            self.current_step = current_step
            self.stats['total_predictions'] += 1
            self.stats['last_update'] = datetime.now().strftime("%H:%M:%S")
            
        except Exception as e:
            print(f"Error in add_buffer_predictions: {e}")

    def get_statistics(self):
        """Get current statistics"""
        return {
            'total_predictions': self.stats['total_predictions'],
            'total_variables': len(self.variable_names),
            'current_step': self.current_step,
            'total_steps': self.total_steps
        }

    def start_server(self, host='127.0.0.1', port=8050, debug=False):
        """Start the Dash server in a separate thread"""
        def run_server():
            self.app.run(host=host, port=port, debug=debug, use_reloader=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        return server_thread

    def clear_data(self):
        """Clear all stored data"""
        self.timestamps.clear()
        for var in self.variable_names:
            if var in self.actual_values:
                self.actual_values[var].clear()
            if var in self.saved_predictions:
                self.saved_predictions[var].clear()
            if var in self.temporal_predictions:
                self.temporal_predictions[var].clear()
        self.stats['total_predictions'] = 0
        self.current_step = 0


# Simplified test function
def test_dashboard():
    """Test function to demonstrate the simplified dashboard"""
    plotter = DashRealTimePlotter()
    
    # Start server
    server_thread = plotter.start_server()
    print("Dashboard started at http://localhost:8050")
    
    # Simulate some data
    variables = ['bits_sent_port1', 'bits_recv_port1', 'bits_sent_port2', 'bits_recv_port2']
    plotter.set_total_steps(100)
    
    for step in range(20):
        # Generate fake predictions and actuals
        predictions = []
        for pred_step in range(6):  # prediction horizon of 6
            predictions.append(np.random.randn(len(variables)) * 10 + 50 + pred_step)
        
        actuals = [predictions[0] + np.random.randn(len(variables)) * 2]
        
        timestamp = datetime.now() + timedelta(seconds=step*5)
        
        plotter.add_buffer_predictions(
            predictions=predictions,
            actuals=actuals,
            current_step=step,
            current_datetime=timestamp,
            variable_names=variables,
            saved_prediction=predictions[0]
        )
        
        time.sleep(1)
    
    print("Test completed. Dashboard should show data.")
    return plotter


if __name__ == "__main__":
    test_dashboard()
    input("Press Enter to stop the server...")
    
    def _get_classification_display(self):
        """Get classification results display - ENHANCED"""
        if not self.classification_results:
            return html.Div("No classification results yet...", 
                          style={'textAlign': 'center', 'color': '#6c757d', 'padding': '20px'})
        
        results = []
        for i, result in enumerate(list(self.classification_results)[-8:]):  # Show last 8
            timestamp, classification = result
            
            # Format classification result nicely
            if isinstance(classification, dict):
                down_ports = classification.get('down_ports', [])
                stormy_ports = classification.get('stormy_ports', [])
                storm_severity = classification.get('storm_severity', 'Unknown')
                description = classification.get('description', '')
                
                # Create status indicators for storms
                if stormy_ports:
                    port_status = f"⚡ Storm Detected on Ports: {stormy_ports}"
                    severity_status = f"🌪️ Severity: {storm_severity}"
                elif down_ports:
                    port_status = f"� Ports Down: {down_ports}"
                    severity_status = "📊 Status: Connectivity Issues"
                else:
                    port_status = "� All Ports Normal"
                    severity_status = "� Status: Normal Traffic"
                
                # Color coding based on storm severity
                if stormy_ports:
                    if storm_severity in ['High', 'Critical', 'Severe']:
                        bg_color = '#ffe6e6'  # Light red for high severity storms
                        border_color = '#dc3545'
                    elif storm_severity in ['Medium', 'Moderate']:
                        bg_color = '#fff3cd'  # Light yellow for medium storms
                        border_color = '#ffc107'
                    else:
                        bg_color = '#d1ecf1'  # Light blue for low storms
                        border_color = '#17a2b8'
                elif down_ports:
                    bg_color = '#ffe6e6'  # Light red for connectivity issues
                    border_color = '#dc3545'
                else:
                    bg_color = '#e6ffe6'  # Light green for normal
                    border_color = '#28a745'
                
                result_content = html.Div([
                    html.Div(f"[{timestamp}]", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    html.Div(port_status, style={'fontSize': '12px'}),
                    html.Div(severity_status, style={'fontSize': '12px'}),
                    html.Div(description, style={'fontSize': '11px', 'fontStyle': 'italic', 'color': '#666'}) if description else None
                ], style={
                    'padding': '8px',
                    'margin': '3px',
                    'backgroundColor': bg_color,
                    'border': f'1px solid {border_color}',
                    'borderRadius': '4px',
                    'fontSize': '12px'
                })
            else:
                # Handle simple string results
                result_content = html.Div(
                    f"[{timestamp}] {classification}",
                    style={'padding': '5px', 'margin': '2px', 'fontSize': '12px'}
                )
            
            results.append(result_content)
        
        return results
    
    def _extract_port_info_from_variables(self, variable_names):
        """Extract port information from variable names"""
        port_info = {}
        
        for var in variable_names:
            if 'Status' in var and 'interface' not in var.lower():
                # Extract port number from "Status 123" format
                try:
                    port_num = var.replace('Status ', '').strip()
                    if port_num.isdigit():
                        port_info[port_num] = {
                            'status_var': var,
                            'traffic_vars': []
                        }
                        
                        # Find corresponding traffic variables
                        for traffic_var in variable_names:
                            if port_num in traffic_var and ('Bits Sent' in traffic_var or 'Bits Received' in traffic_var):
                                port_info[port_num]['traffic_vars'].append(traffic_var)
                except:
                    continue
                    
        return port_info

    def _update_port_status_direct(self, port_statuses, current_time):
        """Update port status using direct port status data from classification - FIXED"""
        try:
            if not hasattr(self, 'port_status'):
                self.port_status = {}
            
            # Clear old statuses to prevent accumulation
            # Keep track of current active ports
            current_ports = set()
            
            # Handle dictionary format (new approach)
            if isinstance(port_statuses, dict):
                for name, status_value in port_statuses.items():
                    if status_value in [1, 2, 1.0, 2.0]:  # Valid status values
                        # Extract port identifier consistently
                        port_id = self._extract_port_id(name)
                        current_ports.add(port_id)
                        
                        # Convert status value to text
                        status_text = "UP" if status_value in [1, 1.0] else "DOWN"
                        
                        # Update existing port or create new one
                        if port_id in self.port_status:
                            # Update existing port
                            self.port_status[port_id]['status'] = status_text
                            self.port_status[port_id]['last_update'] = current_time
                            self.port_status[port_id]['raw_value'] = status_value
                        else:
                            # Create new port entry
                            self.port_status[port_id] = {
                                'status': status_text,
                                'last_update': current_time,
                                'raw_value': status_value,
                                'name': name  # Store original name for reference
                            }
            else:
                # Handle array format (original approach)
                for i, status_value in enumerate(port_statuses):
                    if status_value in [1, 2]:  # Only process valid status values
                        port_id = f"Port_{i+1}"  # Consistent naming
                        current_ports.add(port_id)
                        
                        # Convert status value to text
                        status_text = "UP" if status_value == 1 else "DOWN"
                        
                        # Update existing port or create new one
                        if port_id in self.port_status:
                            # Update existing port
                            self.port_status[port_id]['status'] = status_text
                            self.port_status[port_id]['last_update'] = current_time
                            self.port_status[port_id]['raw_value'] = status_value
                        else:
                            # Create new port entry
                            self.port_status[port_id] = {
                                'status': status_text,
                                'last_update': current_time,
                                'raw_value': status_value,
                                'name': f"Interface {i+1}"
                            }
            
            # Optional: Remove stale ports that are no longer being reported
            # Uncomment this if you want to remove ports that disappear from updates
            # stale_ports = set(self.port_status.keys()) - current_ports
            # for stale_port in stale_ports:
            #     del self.port_status[stale_port]
                        
        except Exception as e:
            print(f"Warning: Port status update failed: {e}")
            # Don't let port status errors break the main prediction display

    def _extract_port_id(self, interface_name):
        """Extract consistent port ID from interface name - NEW HELPER METHOD"""
        try:
            # Handle different interface name formats
            if "gi" in interface_name.lower() and "/" in interface_name:
                # Extract port number like "1/1" from "Interface Gi1/1(name): operational status"
                parts = interface_name.lower().split("gi")[1]
                if "(" in parts:
                    port_id = parts.split("(")[0].strip()
                elif ":" in parts:
                    port_id = parts.split(":")[0].strip()
                else:
                    port_id = parts.strip()
                return f"Gi{port_id}"
            elif "ethernet" in interface_name.lower():
                # Handle Ethernet interfaces
                parts = interface_name.lower().split("ethernet")[1]
                if "(" in parts:
                    port_id = parts.split("(")[0].strip()
                elif ":" in parts:
                    port_id = parts.split(":")[0].strip()
                else:
                    port_id = parts.strip()
                return f"Eth{port_id}"
            else:
                # Fallback: use a hash of the name to ensure consistency
                import hashlib
                hash_obj = hashlib.md5(interface_name.encode())
                return f"Port_{hash_obj.hexdigest()[:6]}"
        except:
            # Ultimate fallback
            return f"Port_{len(self.port_status) + 1}"

    def _update_port_status(self, variable_names, actual_values):
        """Legacy method - now using _update_port_status_direct instead"""
        # This method is deprecated in favor of _update_port_status_direct
        # which uses direct port status data from classification
        pass

    def _get_port_status_display(self):
        """Create port status display with visual indicators - IMPROVED"""
        if not hasattr(self, 'port_status') or not self.port_status:
            return html.Div("No port data available", style={'textAlign': 'center', 'color': '#6c757d'})
        
        port_cards = []
        
        # Sort ports by name for consistent display
        sorted_ports = sorted(self.port_status.items(), key=lambda x: x[0])
        
        for port_id, port_data in sorted_ports:
            status = port_data['status']
            last_update = port_data['last_update']
            raw_value = port_data.get('raw_value', 'N/A')
            original_name = port_data.get('name', port_id)
            
            # Format timestamp
            if hasattr(last_update, 'strftime'):
                last_update_str = last_update.strftime("%H:%M:%S")
            else:
                last_update_str = str(last_update)
            
            # Color coding based on status
            if status == 'UP':
                card_color = '#28a745'  # Green
                text_color = 'white'
                status_icon = "✅"
            elif status == 'DOWN':
                card_color = '#dc3545'  # Red
                text_color = 'white'
                status_icon = "❌"
            else:
                card_color = '#ffc107'  # Yellow
                text_color = 'black'
                status_icon = "❓"
            
            # Create port card with improved layout
            port_card = html.Div([
                html.Div([
                    html.H5(port_id, style={'margin': '0 0 5px 0', 'color': text_color, 'fontSize': '14px'}),
                    html.P(f"{status} {status_icon}", style={'margin': '0 0 5px 0', 'fontWeight': 'bold', 'color': text_color}),
                    html.Small(f"Value: {raw_value}", style={'color': text_color, 'opacity': '0.8', 'display': 'block'}),
                    html.Small(f"{last_update_str}", style={'color': text_color, 'opacity': '0.8', 'display': 'block'})
                ])
            ], style={
                'backgroundColor': card_color,
                'padding': '8px',
                'borderRadius': '5px',
                'minWidth': '110px',
                'maxWidth': '130px',
                'textAlign': 'center',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'margin': '3px',
                'flex': '0 0 auto'
            })
            
            port_cards.append(port_card)
        
        return port_cards
        
        return port_cards

    def _get_alerts_display(self):
        """Get recent storm alerts display - FOCUSED ON STORMS"""
        if not hasattr(self, 'classification_results') or not self.classification_results:
            return html.Div("No storm alerts", style={'textAlign': 'center', 'color': '#6c757d'})
        
        # Extract storm alerts from recent classification results
        storm_alerts = []
        for timestamp, classification in list(self.classification_results)[-10:]:  # Check last 10
            if isinstance(classification, dict):
                stormy_ports = classification.get('stormy_ports', [])
                storm_severity = classification.get('storm_severity', '')
                down_ports = classification.get('down_ports', [])
                
                # Create alert if storms detected
                if stormy_ports:
                    severity_color = '#dc3545' if storm_severity in ['High', 'Critical', 'Severe'] else '#ffc107'
                    alert_item = html.Div([
                        html.Span(f"⚡ {timestamp}", style={'fontWeight': 'bold', 'color': severity_color}),
                        html.Br(),
                        html.Span(f"Storm on ports: {', '.join(map(str, stormy_ports))}", style={'fontSize': '11px'}),
                        html.Br(),
                        html.Span(f"Severity: {storm_severity}", style={'fontSize': '10px', 'fontStyle': 'italic'})
                    ], style={
                        'padding': '4px',
                        'margin': '2px',
                        'border': f'1px solid {severity_color}',
                        'borderRadius': '3px',
                        'backgroundColor': '#fff3cd' if storm_severity not in ['High', 'Critical', 'Severe'] else '#ffe6e6'
                    })
                    storm_alerts.append(alert_item)
                elif down_ports:
                    # Show connectivity alerts too
                    alert_item = html.Div([
                        html.Span(f"🔴 {timestamp}", style={'fontWeight': 'bold', 'color': '#dc3545'}),
                        html.Br(),
                        html.Span(f"Ports down: {', '.join(map(str, down_ports))}", style={'fontSize': '11px'})
                    ], style={
                        'padding': '4px',
                        'margin': '2px',
                        'border': '1px solid #dc3545',
                        'borderRadius': '3px',
                        'backgroundColor': '#ffe6e6'
                    })
                    storm_alerts.append(alert_item)
        
        if not storm_alerts:
            return html.Div("No active storm alerts", 
                          style={'textAlign': 'center', 'color': '#28a745', 'padding': '10px'})
        
        return storm_alerts[-5:]  # Show last 5 storm alerts only
    
    def set_total_steps(self, total_steps):
        """Set the total number of steps for progress tracking"""
        self.total_steps = total_steps
        print(f"Dashboard: Set total steps to {total_steps}")
    
    def add_buffer_predictions(self, predictions, actuals, current_step, current_datetime, variable_names, saved_prediction=None, future_prediction=None, port_statuses=None, classification_result=None):
        """
        Add new prediction data to the dashboard - simplified version
        
        Args:
            predictions: List of prediction arrays for temporal horizon (t+1 to t+6)
            actuals: List of actual arrays for each time step  
            current_step: Current step number
            current_datetime: Current timestamp
            variable_names: List of variable names
            saved_prediction: Single prediction array for t+1
            future_prediction: Not used in simplified version
            port_statuses: Array of current port status values from classification data
            classification_result: Classification prediction result (dict with down_ports, etc.)
        """
        try:
            # Initialize variable names if first time
            if not self.variable_names:
                self.variable_names = variable_names
                print(f"Dashboard: Initializing {len(variable_names)} variables")
                for var in variable_names:
                    self.actual_values[var] = deque(maxlen=self.max_points)
                    self.saved_predictions[var] = deque(maxlen=self.max_points)
                    self.temporal_predictions[var] = deque(maxlen=10)  # Keep recent temporal predictions
            
            # Update port status when new data arrives - using direct port status data
            if port_statuses is not None:
                self._update_port_status_direct(port_statuses, current_datetime)
            
            # Store classification result
            if classification_result is not None:
                self.add_classification_result(classification_result, current_datetime)
            
            # Validate inputs
            if not predictions or not actuals:
                print("Dashboard: Empty predictions or actuals received")
                return
            
            if len(predictions) == 0 or len(actuals) == 0:
                print("Dashboard: No prediction or actual data")
                return
                
            # Use saved prediction if provided, otherwise use first prediction
            pred_t1_step = saved_prediction if saved_prediction is not None else predictions[0]
            actual_step = actuals[0]
            
            # Validate data lengths
            if len(pred_t1_step) != len(variable_names) or len(actual_step) != len(variable_names):
                print(f"Dashboard: Data length mismatch. Pred: {len(pred_t1_step)}, Actual: {len(actual_step)}, Variables: {len(variable_names)}")
                return
            
            # Add timestamp
            self.timestamps.append(current_datetime)
            
            # Add data for each variable - SIMPLIFIED
            for i, var in enumerate(variable_names):
                # 1. Actual values
                self.actual_values[var].append(actual_step[i])
                
                # 2. Saved predictions (t+1)
                self.saved_predictions[var].append(pred_t1_step[i])
                
                # 3. Temporal predictions - SIMPLIFIED storage
                if predictions and len(predictions) > 0:
                    # Store the full prediction horizon for this variable
                    var_predictions = []
                    for pred_step in predictions:
                        if i < len(pred_step):
                            var_predictions.append(pred_step[i])
                    
                    if var_predictions:
                        self.temporal_predictions[var].append(var_predictions)
            
            # Update statistics
            self.current_step = current_step
            self.stats['total_predictions'] += 1
            self.stats['last_update'] = datetime.now().strftime("%H:%M:%S")
            
            # Debug info - simplified
            if current_step % 10 == 0:  # Print every 10 steps
                print(f"Dashboard: Step {current_step}, Variables: {len(variable_names)}, "
                      f"Predictions: {len(predictions)}, Actuals: {len(actuals)}")
            
        except Exception as e:
            print(f"Error in add_buffer_predictions: {e}")
            import traceback
            traceback.print_exc()
    
    def add_classification_result(self, classification_result, timestamp=None):
        """Add classification result to display"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Format timestamp
        if hasattr(timestamp, 'strftime'):
            timestamp_str = timestamp.strftime("%H:%M:%S")
        else:
            timestamp_str = str(timestamp)
        
        # Store the classification result
        self.classification_results.append((timestamp_str, classification_result))
        print(f"Dashboard: Classification result: {classification_result}")
    
    def get_statistics(self):
        """Get current statistics"""
        return {
            'total_predictions': self.stats['total_predictions'],
            'total_variables': len(self.variable_names),
            'prediction_errors': dict(self.stats['prediction_errors']),
            'current_step': self.current_step,
            'total_steps': self.total_steps
        }
    
    def start_server(self, host='127.0.0.1', port=8050, debug=False):
        """Start the Dash server in a separate thread"""
        def run_server():
            self.app.run(host=host, port=port, debug=debug, use_reloader=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        return server_thread
    
    def clear_data(self):
        """Clear all stored data"""
        self.timestamps.clear()
        for var in self.variable_names:
            if var in self.actual_values:
                self.actual_values[var].clear()
            if var in self.saved_predictions:
                self.saved_predictions[var].clear()
            if var in self.temporal_predictions:
                self.temporal_predictions[var].clear()
        self.classification_results.clear()
        self.stats['total_predictions'] = 0
        self.current_step = 0
        print("Dashboard: Data cleared")


# Example usage function for testing
def test_dashboard():
    """Test function to demonstrate the dashboard"""
    plotter = DashRealTimePlotter()
    
    # Start server
    server_thread = plotter.start_server()
    print("Dashboard started at http://localhost:8050")
    
    # Simulate some data
    variables = ['bits_sent_port1', 'bits_recv_port1', 'bits_sent_port2', 'bits_recv_port2']
    plotter.set_total_steps(100)
    
    for step in range(20):
        # Generate fake predictions and actuals
        # Generate multiple prediction steps (t+1 to t+6)
        predictions = []
        for pred_step in range(6):  # prediction horizon of 6
            predictions.append(np.random.randn(len(variables)) * 10 + 50 + pred_step)
        
        actuals = [predictions[0] + np.random.randn(len(variables)) * 2]
        
        timestamp = datetime.now() + timedelta(seconds=step*5)
        
        plotter.add_buffer_predictions(
            predictions=predictions,
            actuals=actuals,
            current_step=step,
            current_datetime=timestamp,
            variable_names=variables,
            saved_prediction=predictions[0]  # Pass the saved prediction (t+1)
        )
        
        # Add some classification results
        if step % 5 == 0:
            plotter.add_classification_result(f"Normal Operation {step}")
        
        time.sleep(1)  # Wait 1 second between updates
    
    print("Test completed. Dashboard should show data.")
    return plotter


if __name__ == "__main__":
    # Run test if this file is executed directly
    test_dashboard()
    input("Press Enter to stop the server...")
