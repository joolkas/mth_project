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
    def __init__(self, max_points=60, update_interval=60000):
        """
        Simplified real-time plotter for industrial network forecasting data.
        
        Args:
            max_points: Maximum number of points to display in each plot
            update_interval: Update interval in milliseconds (default 60 seconds)
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
        self.classification_results = deque(maxlen=50)  # Re-added for classification display
        self.label_to_name_dict = {}      # Store label_to_name dictionary
        
        # Port status tracking - Re-added
        self.port_status = {}
        
        # Basic statistics
        self.stats = {
            'total_predictions': 0,
            'last_update': None
        }
        
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup organized Dash app layout with classification, sections, and port status"""
        self.app.layout = html.Div([
            # Classification results at the top
            html.Div([
                html.H2("Classification Results", style={'textAlign': 'center', 'color': '#e74c3c', 'margin': '10px'}),
                html.Div(id='classification-results', style={
                    'padding': '15px', 
                    'backgroundColor': '#fff5f5', 
                    'borderRadius': '5px',
                    'border': '1px solid #e74c3c',
                    'margin': '10px',
                    'textAlign': 'center'
                })
            ]),
            
            # Main Header
            html.Div([
                html.H1("Network Traffic Forecasting Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px'})
            ]),
            
            # Basic Status Info
            html.Div(id='status-info', 
                    style={'textAlign': 'center', 'fontSize': '14px', 'margin': '20px'}),
            
            # System Metrics section (CPU, Memory, etc.)
            html.Div([
                html.H3("Numeric Metrics", style={'textAlign': 'center', 'color': '#3498db', 'margin': '30px 20px'}),
                html.Div(id='system-graphs-container', style={'margin': '10px'})
            ], style={'marginBottom': '40px'}),
            
            # Port sections (Bits Sent/Received)
            html.Div([
                html.H3("Port Traffic (Bits Sent/Received)", style={'textAlign': 'center', 'color': '#9b59b6', 'margin': '30px 20px'}),
                html.Div(id='port-graphs-container', style={'margin': '20px'})
            ], style={'marginBottom': '40px'}),
            
            # Port status display removed - causing issues
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            )
        ])
    
    def _setup_callbacks(self):
        """Setup organized Dash callbacks for classification, sections, and port status"""
        @self.app.callback(
            [Output('classification-results', 'children'),
             Output('status-info', 'children'),
             Output('system-graphs-container', 'children'),
             Output('port-graphs-container', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            return (
                self._get_classification_results(),
                self._get_status_info(),
                self._update_system_graphs(),
                self._update_port_graphs()
            )
    
    def _categorize_variables(self):
        """Categorize variables into numeric metrics and bits sent/received by port"""
        numeric_vars = []
        port_bits_vars = {}
        
        for var in self.variable_names:
            var_lower = var.lower()
            
            # Check if this is bits sent/received
            if 'bits' in var_lower and ('sent' in var_lower or 'recv' in var_lower or 'received' in var_lower):
                # Extract port identifier from bits variables (e.g., 1/0/1, 1/0/2, etc.)
                port_id = self._extract_port_id_from_bits(var)
                if port_id not in port_bits_vars:
                    port_bits_vars[port_id] = []
                port_bits_vars[port_id].append(var)
            else:
                # All other numeric values go to numeric section
                numeric_vars.append(var)
        
        return numeric_vars, port_bits_vars
    
    def _extract_port_id_from_bits(self, variable_name):
        """Extract port identifier from bits variable names (e.g., 1/0/1, 1/0/2)"""
        import re
        
        # Look for patterns like 1/0/1, 1/0/2, etc. in variable names
        match = re.search(r'(\d+/\d+/\d+)', variable_name)
        if match:
            return f"Port {match.group(1)}"
        
        # Look for simpler port patterns
        match = re.search(r'port[_\-]?(\d+)|(\d+)[_\-]?port', variable_name.lower())
        if match:
            return f"Port {match.group(1) or match.group(2)}"
        
        # Look for just numbers in variable names
        numbers = re.findall(r'\d+', variable_name)
        if numbers:
            return f"Port {numbers[-1]}"  # Use last number as port ID
        
        return f"Port {variable_name}"
    
    def _extract_port_id(self, variable_name):
        """Extract port identifier from variable name"""
        import re
        # Look for port patterns like "port1", "Port_1", "1_port", etc.
        match = re.search(r'port[_\-]?(\d+)|(\d+)[_\-]?port', variable_name.lower())
        if match:
            return f"Port {match.group(1) or match.group(2)}"
        
        # Look for just numbers in variable names
        numbers = re.findall(r'\d+', variable_name)
        if numbers:
            return f"Port {numbers[0]}"
        
        return f"Port {variable_name}"
    
    def _get_classification_results(self):
        """Display recent classification results and label_to_name dictionary"""
        try:
            content_items = []
            
            # Display label_to_name dictionary if available
            if hasattr(self, 'label_to_name_dict') and self.label_to_name_dict:
                content_items.append(
                    html.Div([
                        html.H4("Classification Labels:", style={'color': '#2c3e50', 'marginBottom': '10px'}),
                        html.Div([
                            html.Div(f"{idx}: {name}", style={'margin': '2px', 'padding': '3px', 'backgroundColor': '#f8f9fa', 'borderRadius': '3px'})
                            for idx, name in self.label_to_name_dict.items()
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '5px'})
                    ], style={'marginBottom': '15px'})
                )
            
            # Display recent classification results
            if self.classification_results:
                recent_results = list(self.classification_results)[-5:]  # Show last 5 results
                result_items = []
                
                for i, result in enumerate(recent_results):
                    timestamp = result.get('timestamp', 'N/A')
                    classification = result.get('classification', 'Unknown')
                    confidence = result.get('confidence', 0)
                    
                    color = '#e74c3c' if 'storm' in str(classification).lower() else '#27ae60'
                    
                    result_items.append(
                        html.Div([
                            html.Span(f"Time: {timestamp} | ", style={'fontWeight': 'bold'}),
                            html.Span(f"Result: {classification} ", style={'color': color, 'fontWeight': 'bold'}),
                            html.Span(f"(Confidence: {confidence:.2f})")
                        ], style={'margin': '5px'})
                    )
                
                content_items.append(
                    html.Div([
                        html.H4("Recent Results:", style={'color': '#2c3e50', 'marginBottom': '10px'}),
                        html.Div(result_items)
                    ])
                )
            
            if not content_items:
                return html.Div("No classification information available", 
                              style={'textAlign': 'center', 'color': '#999'})
            
            return html.Div(content_items)
        except Exception as e:
            return html.Div(f"Error displaying classification results: {str(e)}", 
                           style={'color': '#e74c3c'})
    
    def _update_system_graphs(self):
        """Update graphs for numeric metrics (all except bits sent/received)"""
        try:
            if not self.variable_names:
                return html.Div("Waiting for data...", style={'textAlign': 'center', 'padding': '50px'})
            
            numeric_vars, _ = self._categorize_variables()
            
            if not numeric_vars:
                return html.Div("No numeric metrics available", style={'textAlign': 'center', 'padding': '20px'})
            
            return self._create_section_graph(numeric_vars, "Numeric Metrics")
        except Exception as e:
            return html.Div(f"Error updating numeric graphs: {str(e)}", style={'color': '#e74c3c'})
    
    def _update_port_graphs(self):
        """Update graphs organized by port bits (sent/received pairs)"""
        try:
            if not self.variable_names:
                return html.Div("Waiting for data...", style={'textAlign': 'center', 'padding': '50px'})
            
            _, port_bits_vars = self._categorize_variables()
            
            if not port_bits_vars:
                return html.Div("No port bits metrics available", style={'textAlign': 'center', 'padding': '20px'})
            
            port_sections = []
            for port_id, variables in port_bits_vars.items():
                section_graph = self._create_section_graph(variables, f"{port_id} - Bits Sent/Received")
                port_sections.append(
                    html.Div([
                        html.H4(f"{port_id} - Bits Sent/Received", style={'textAlign': 'center', 'color': '#9b59b6', 'margin': '20px'}),
                        section_graph
                    ], style={'marginBottom': '30px'})
                )
            
            return html.Div(port_sections)
        except Exception as e:
            return html.Div(f"Error updating port bits graphs: {str(e)}", style={'color': '#e74c3c'})
    
    def _create_section_graph(self, variables, section_title):
        """Create a graph for a specific section of variables"""
        try:
            if not variables:
                return html.Div(f"No variables for {section_title}", style={'textAlign': 'center'})
            
            # Create two-column layout for the section
            num_vars = len(variables)
            cols = 2
            rows = (num_vars + cols - 1) // cols
            
            # Increase spacing for better visibility with more space above plots
            vertical_spacing = 0.08  # Increased from 0.05 for more space above each plot
            horizontal_spacing = 0.03
            
            # Simple subplot titles
            subplot_titles = [var for var in variables]
            
            fig = sp.make_subplots(
                rows=rows, 
                cols=cols,
                subplot_titles=subplot_titles,
                vertical_spacing=vertical_spacing,
                horizontal_spacing=horizontal_spacing
            )
            
            # Add traces for each variable in this section
            for i, var in enumerate(variables):
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
                
                # 2. Saved predictions (red) - keep historical only
                if var in self.saved_predictions and len(self.saved_predictions[var]) > 0:
                    pred_data = list(self.saved_predictions[var])
                    data_length = min(len(pred_data), len(timestamps))
                    
                    if data_length > 0:
                        recent_timestamps = timestamps[-data_length:]
                        recent_pred_data = pred_data[-data_length:]
                        
                        if len(recent_timestamps) > 0 and len(recent_pred_data) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=recent_timestamps,
                                    y=recent_pred_data,
                                    mode='lines+markers',
                                    name='Saved Predictions',
                                    line=dict(color='red', width=2, dash='dash'),
                                    marker=dict(size=4),
                                    showlegend=(i == 0)
                                ),
                                row=row, col=col
                            )
                
                # 3. Temporal predictions (green) - Future predictions t+1 to t+6 connected to red plot
                if var in self.temporal_predictions and len(self.temporal_predictions[var]) > 0:
                    temporal_data = list(self.temporal_predictions[var])
                    
                    if timestamps and len(temporal_data) > 0:
                        current_time = timestamps[-1] if timestamps else datetime.now()
                        
                        # Get the latest temporal prediction (should be array of 6 future values)
                        latest_temporal = temporal_data[-1] if temporal_data else []
                        
                        if isinstance(latest_temporal, list) and len(latest_temporal) > 0:
                            # Start green plot from current time (t) and go through t+5
                            future_timestamps = []
                            future_predictions = []
                            
                            # Add future timestamps and predictions starting from current time (t)
                            for step in range(0, min(6, len(latest_temporal))):
                                future_time = current_time + timedelta(minutes=step)  # t, t+1, t+2, ..., t+5
                                future_timestamps.append(future_time)
                                future_predictions.append(latest_temporal[step])
                            
                            if len(future_timestamps) > 0 and len(future_predictions) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=future_timestamps,
                                        y=future_predictions,
                                        mode='lines+markers',
                                        name='Future Predictions (t to t+5)',
                                        line=dict(color='green', width=2, dash='dot'),
                                        marker=dict(size=3),
                                        showlegend=(i == 0)
                                    ),
                                    row=row, col=col
                                )
            
            # Update layout with more space above plots and below legend
            fig.update_layout(
                height=380 * rows,  # Increased height for more legend space
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.08,  # Moved legend even higher for more space below
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=110, b=80)  # More top and bottom margins
            )
            
            # Remove x-axis labels from all but bottom row
            for i in range(1, rows + 1):
                for j in range(1, cols + 1):
                    if i < rows:
                        fig.update_xaxes(showticklabels=False, row=i, col=j)
            
            return dcc.Graph(figure=fig, style={'height': f'{380 * rows}px'})  # Updated height with more legend space
            
        except Exception as e:
            return html.Div(f"Error creating {section_title} graph: {str(e)}", style={'color': '#e74c3c'})
    
    def _get_port_status_display(self):
        """Display current port status at the bottom"""
        try:
            if not self.port_status:
                return html.Div("No port status information available", 
                              style={'textAlign': 'center', 'color': '#999'})
            
            status_items = []
            for port_id, status_info in self.port_status.items():
                status = status_info.get('status', 'Unknown')
                last_update = status_info.get('last_update', 'N/A')
                storm_detected = status_info.get('storm_detected', False)
                raw_value = status_info.get('raw_value', 'N/A')
                original_name = status_info.get('original_name', port_id)
                
                # Color coding based on status
                if storm_detected:
                    color = '#e74c3c'  # Red for storm
                    bg_color = '#fff5f5'
                elif status == 'Active':
                    color = '#27ae60'  # Green for active
                    bg_color = '#f0fff4'
                else:
                    color = '#f39c12'  # Orange for other states
                    bg_color = '#fffbf0'
                
                status_items.append(
                    html.Div([
                        html.Div([
                            html.Strong(f"{original_name}:", style={'color': color, 'fontSize': '14px'}),
                            html.Br(),
                            html.Span(f"Status: {status} ({raw_value})", style={'marginLeft': '5px', 'fontSize': '12px'}),
                            html.Br(),
                            html.Span(f"Updated: {last_update}", style={'marginLeft': '5px', 'fontSize': '11px', 'color': '#666'}),
                            html.Br() if storm_detected else "",
                            html.Span("⚠️ STORM DETECTED", style={'marginLeft': '5px', 'color': '#e74c3c', 'fontWeight': 'bold', 'fontSize': '11px'}) if storm_detected else ""
                        ])
                    ], style={
                        'display': 'inline-block', 
                        'margin': '8px', 
                        'padding': '12px',
                        'backgroundColor': bg_color,
                        'borderRadius': '8px',
                        'border': f'2px solid {color}',
                        'minWidth': '180px',
                        'maxWidth': '220px',
                        'verticalAlign': 'top'
                    })
                )
            
            return html.Div(status_items, style={'textAlign': 'center', 'padding': '10px'})
        except Exception as e:
            return html.Div(f"Error displaying port status: {str(e)}", style={'color': '#e74c3c'})
    
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
        Add new prediction data to the dashboard - enhanced with classification and port status
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
            
            # Handle classification results
            if classification_result is not None:
                timestamp_str = current_datetime.strftime("%H:%M:%S")
                
                # If classification_result is the label_to_name dictionary, store it
                if isinstance(classification_result, dict) and all(isinstance(k, (int, str)) and isinstance(v, str) for k, v in classification_result.items()):
                    self.label_to_name_dict = classification_result
                    # Don't add this as a classification result, it's just the dictionary
                elif isinstance(classification_result, dict):
                    # This is an actual classification result
                    classification_entry = {
                        'timestamp': timestamp_str,
                        'classification': classification_result.get('classification', 'Unknown'),
                        'confidence': classification_result.get('confidence', 0.0)
                    }
                    self.classification_results.append(classification_entry)
                else:
                    # Handle simple format
                    classification_entry = {
                        'timestamp': timestamp_str,
                        'classification': str(classification_result),
                        'confidence': 1.0
                    }
                    self.classification_results.append(classification_entry)
            
            # Handle port statuses - Show ALL ports passed in port_statuses
            if port_statuses is not None:
                current_time_str = current_datetime.strftime("%H:%M:%S")
                if isinstance(port_statuses, dict):
                    for port_name, status_value in port_statuses.items():
                        # Use original port name for better identification
                        port_display_name = port_name if port_name else self._extract_port_id(port_name)
                        
                        # Determine status based on value
                        status_text = "Active" if status_value in [1, 1.0, True] else "Inactive"
                        storm_detected = False
                        
                        # Check if this is a storm indication
                        if classification_result and isinstance(classification_result, dict):
                            if 'storm' in str(classification_result.get('classification', '')).lower():
                                storm_detected = True
                        
                        # Store with original port name for complete information
                        self.port_status[port_display_name] = {
                            'status': status_text,
                            'last_update': current_time_str,
                            'storm_detected': storm_detected,
                            'raw_value': status_value,
                            'original_name': port_name
                        }
            
            # Update statistics
            self.current_step = current_step
            self.stats['total_predictions'] += 1
            self.stats['last_update'] = datetime.now().strftime("%H:%M:%S")
            
        except Exception as e:
            print(f"Error in add_buffer_predictions: {e}")
    
    def set_label_to_name_dict(self, label_to_name_dict):
        """Set the label_to_name dictionary for classification display"""
        if isinstance(label_to_name_dict, dict):
            self.label_to_name_dict = label_to_name_dict
            print(f"Dashboard: Set label_to_name dictionary with {len(label_to_name_dict)} labels")

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
    # test_dashboard()
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


# if __name__ == "__main__":
#     # Run test if this file is executed directly
#     test_dashboard()
#     input("Press Enter to stop the server...")
