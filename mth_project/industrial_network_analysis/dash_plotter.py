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
import queue


class DashRealTimePlotter:
    def __init__(self, max_points=60, update_interval=1000):
        """
        Real-time plotter for industrial network forecasting data.
        
        Timing convention:
        - Data points are 1 minute apart
        - If current time is 06:30:00, then:
          - Blue line (actual): shows historical values up to 06:29:00 (excludes last sample)
          - Red line (t+1): shows predictions aligned with historical timestamps
          - Orange line (t+6): starts from 06:29:00 and shows 06:30:00, 06:31:00, 06:32:00, 06:33:00, 06:34:00, 06:35:00
        
        Args:
            max_points: Maximum number of points to display in each plot
            update_interval: Update interval in milliseconds
        """
        self.app = dash.Dash(__name__)
        self.max_points = max_points
        self.update_interval = update_interval
        
        # Data storage - simplified to three essential streams
        self.data_queue = queue.Queue()
        self.timestamps = deque(maxlen=max_points)
        self.actual_values = {}           # 1. Actual values (historical)
        self.saved_predictions = {}       # 2. Saved predictions with latest t+1
        self.temporal_predictions = {}    # 3. Temporal predictions (t+6 horizon)
        self.variable_names = []
        self.current_step = 0
        self.total_steps = 0
        self.classification_results = deque(maxlen=50)
        self.prediction_horizon = 6
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'last_update': None,
            'prediction_errors': {}
        }
        
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup the Dash app layout with 3 columns maximum"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Industrial Network Forecasting - Real-Time Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.Div(id='status-info', 
                        style={'textAlign': 'center', 'fontSize': '16px', 'margin': '10px'})
            ]),
            
            # Main content with graphs
            html.Div(id='graphs-container'),
            
            # Classification results
            html.Div([
                html.H3("Recent Classification Results", style={'color': '#34495e'}),
                html.Div(id='classification-results', style={'height': '100px', 'overflow': 'auto'})
            ], style={'margin': '20px', 'padding': '10px', 'border': '1px solid #bdc3c7'}),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            )
        ])
    
    def _setup_callbacks(self):
        """Setup Dash callbacks for real-time updates"""
        @self.app.callback(
            [Output('graphs-container', 'children'),
             Output('status-info', 'children'),
             Output('classification-results', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            return self._update_graphs(), self._get_status_info(), self._get_classification_display()
    
    def _update_graphs(self):
        """Update all graphs with latest data - simplified to show three traces per variable"""
        try:
            if not self.variable_names:
                return html.Div("Waiting for data...", style={'textAlign': 'center', 'padding': '50px'})
            
            # Create two-column layout
            num_vars = len(self.variable_names)
            cols = 2
            rows = (num_vars + cols - 1) // cols
            
            # print(f"Dashboard Debug: {num_vars} variables, {cols} columns, {rows} rows")
            
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
            
            # Add traces for each variable - simplified to three traces only
            for i, var in enumerate(self.variable_names):
                row = i // cols + 1
                col = i % cols + 1
                
                timestamps = list(self.timestamps)
                
                # 1. Actual values trace (blue) - exclude last sample
                if var in self.actual_values and len(self.actual_values[var]) > 0:
                    actual_data = list(self.actual_values[var])
                    data_length = min(len(actual_data), len(timestamps))
                    
                    if data_length > 1:  # Need at least 2 points to remove last one
                        # Remove the last sample from display
                        recent_timestamps = timestamps[-data_length:-1]  # Exclude last timestamp
                        recent_actual_data = actual_data[-data_length:-1]  # Exclude last actual value
                        
                        if len(recent_timestamps) > 0 and len(recent_actual_data) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=recent_timestamps,
                                    y=recent_actual_data,
                                    mode='lines+markers',
                                    name='Actual Values',
                                    line=dict(color='blue', width=2),
                                    marker=dict(size=4),
                                    showlegend=(i == 0),
                                    hovertemplate='<b>Actual Values</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
                                ),
                                row=row, col=col
                            )
                
                # 2. Saved predictions (red) - no time extension
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
                                name='Saved Pred (t+1)',
                                line=dict(color='red', width=2, dash='dash'),
                                marker=dict(size=4),
                                showlegend=(i == 0),
                                hovertemplate='<b>Saved Pred (t+1)</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
                            ),
                            row=row, col=col
                        )
                
                # 3. Temporal predictions t+6 horizon (orange) - shorter and moved one sample left
                if var in self.temporal_predictions and len(self.temporal_predictions[var]) > 0:
                    temporal_data = list(self.temporal_predictions[var])
                    
                    if timestamps and len(temporal_data) > 0 and len(timestamps) > 1:
                        # Start from one sample before the current time (moved left)
                        start_time = timestamps[-2] if len(timestamps) >= 2 else timestamps[-1]
                        
                        # Create shorter orange line - start from one sample left, no connection point
                        future_timestamps = []
                        future_predictions = []
                        
                        # Add future predictions t+1 to t+6 (no t=0 connection point)
                        for step in range(1, min(7, len(temporal_data[-1]) + 1)):
                            future_time = start_time + timedelta(minutes=step)
                            future_timestamps.append(future_time)
                            
                            # Use the most recent temporal prediction array
                            if step <= len(temporal_data[-1]):
                                future_predictions.append(temporal_data[-1][step - 1])
                            else:
                                future_predictions.append(temporal_data[-1][-1])
                        
                        if len(future_timestamps) > 0 and len(future_predictions) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=future_timestamps,
                                    y=future_predictions,
                                    mode='lines+markers',
                                    name='Temporal Pred (t+6)',
                                    line=dict(color='orange', width=2, dash='dot'),
                                    marker=dict(size=3),
                                    showlegend=(i == 0),
                                    hovertemplate='<b>Temporal Pred (t+6)</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
                                ),
                                row=row, col=col
                            )
            
            # Simple layout
            total_height = 800 * rows
            
            fig.update_layout(
                height=total_height,
                title_text="Real-Time Forecasting Dashboard",
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
            print(f"Error in _update_graphs: {e}")
            import traceback
            traceback.print_exc()
            return html.Div(f"Error updating graphs: {str(e)}", 
                          style={'textAlign': 'center', 'padding': '50px', 'color': 'red'})
    
    def _get_status_info(self):
        """Get current status information"""
        progress = (self.current_step / max(self.total_steps, 1)) * 100 if self.total_steps > 0 else 0
        
        return html.Div([
            html.Span(f"Progress: {self.current_step}/{self.total_steps} ({progress:.1f}%) | "),
            html.Span(f"Total Predictions: {self.stats['total_predictions']} | "),
            html.Span(f"Variables: {len(self.variable_names)} | "),
            html.Span(f"Last Update: {self.stats['last_update'] or 'Never'}")
        ])
    
    def _get_classification_display(self):
        """Get classification results display"""
        if not self.classification_results:
            return "No classification results yet..."
        
        results = []
        for i, result in enumerate(list(self.classification_results)[-10:]):  # Show last 10
            timestamp, classification = result
            results.append(
                html.Div(f"{timestamp}: {classification}", 
                        style={'margin': '2px', 'padding': '2px'})
            )
        
        return results
    
    def set_total_steps(self, total_steps):
        """Set the total number of steps for progress tracking"""
        self.total_steps = total_steps
        print(f"Dashboard: Set total steps to {total_steps}")
    
    def add_buffer_predictions(self, predictions, actuals, current_step, current_datetime, variable_names, saved_prediction=None, future_prediction=None):
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
            
            # Add data for each variable - simplified to three streams
            for i, var in enumerate(variable_names):
                # 1. Actual values
                self.actual_values[var].append(actual_step[i])
                
                # 2. Saved predictions (t+1)
                self.saved_predictions[var].append(pred_t1_step[i])
                
                # 3. Temporal predictions (t+6 horizon) - store the full prediction horizon
                var_temporal_predictions = [pred[i] for pred in predictions if i < len(pred)]
                if var_temporal_predictions:
                    self.temporal_predictions[var].append(var_temporal_predictions)
            
            # Update statistics
            self.current_step = current_step
            self.stats['total_predictions'] += 1
            self.stats['last_update'] = datetime.now().strftime("%H:%M:%S")
            
            print(f"Dashboard: Added data for step {current_step}, {len(variable_names)} variables")
            
        except Exception as e:
            print(f"Error in add_buffer_predictions: {e}")
            import traceback
            traceback.print_exc()
    
    def add_classification_result(self, classification_result):
        """Add classification result to display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.classification_results.append((timestamp, classification_result))
        print(f"Dashboard: Added classification result: {classification_result}")
    
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
    variables = ['temperature', 'pressure', 'flow_rate', 'cpu_usage']
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
