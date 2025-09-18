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
    def __init__(self, max_points=200, update_interval=1000):
        """
        Real-time plotter for industrial network forecasting data.
        
        Args:
            max_points: Maximum number of points to display in each plot
            update_interval: Update interval in milliseconds
        """
        self.app = dash.Dash(__name__)
        self.max_points = max_points
        self.update_interval = update_interval
        
        # Data storage
        self.data_queue = queue.Queue()
        self.timestamps = deque(maxlen=max_points)
        self.predictions_data = {}
        self.actuals_data = {}
        self.variable_names = []
        self.current_step = 0
        self.total_steps = 0
        self.classification_results = deque(maxlen=50)
        
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
        """Update all graphs with latest data"""
        try:
            if not self.variable_names:
                return html.Div("Waiting for data...", style={'textAlign': 'center', 'padding': '50px'})
            
            # Create subplots organized in max 3 columns
            num_vars = len(self.variable_names)
            cols = min(3, num_vars)
            rows = (num_vars + cols - 1) // cols  # Ceiling division
            
            print(f"Dashboard Debug: {num_vars} variables, {cols} columns, {rows} rows")
            
            # Calculate appropriate spacing based on number of rows
            # Maximum vertical spacing is 1/(rows-1), we use 80% of that to be safe
            if rows > 1:
                max_vertical_spacing = 1.0 / (rows - 1)
                vertical_spacing = min(0.08, max_vertical_spacing * 0.8)
                print(f"Dashboard Debug: Using vertical spacing {vertical_spacing:.4f} (max allowed: {max_vertical_spacing:.4f})")
            else:
                vertical_spacing = 0.08
            
            # Create subplot titles with line breaks for long names
            subplot_titles = []
            for var in self.variable_names:
                # Break long variable names into two lines
                if len(var) > 25:  # If variable name is longer than 25 characters
                    # Try to break at a natural point (space, underscore, or dash)
                    break_points = [' ', '_', '-', '.']
                    mid_point = len(var) // 2
                    best_break = mid_point
                    
                    # Find the best break point near the middle
                    for i in range(max(0, mid_point - 10), min(len(var), mid_point + 10)):
                        if var[i] in break_points:
                            best_break = i
                            break
                    
                    line1 = var[:best_break].strip()
                    line2 = var[best_break:].strip()
                    title = f"{line1}<br>{line2} - Pred vs Actual"
                else:
                    title = f"{var}<br>Predictions vs Actuals"
                subplot_titles.append(title)
            
            fig = sp.make_subplots(
                rows=rows, 
                cols=cols,
                subplot_titles=subplot_titles,
                vertical_spacing=vertical_spacing,
                horizontal_spacing=0.06
            )
            
            # Add traces for each variable
            for i, var in enumerate(self.variable_names):
                row = i // cols + 1
                col = i % cols + 1
                
                if var in self.predictions_data and var in self.actuals_data:
                    # Get data for this variable
                    pred_data = list(self.predictions_data[var])
                    actual_data = list(self.actuals_data[var])
                    timestamps = list(self.timestamps)
                    
                    if len(pred_data) > 0 and len(actual_data) > 0:
                        # Ensure we have corresponding timestamps
                        data_length = min(len(pred_data), len(actual_data), len(timestamps))
                        if data_length > 0:
                            # Get the most recent data points
                            recent_timestamps = timestamps[-data_length:]
                            recent_pred_data = pred_data[-data_length:]
                            recent_actual_data = actual_data[-data_length:]
                            
                            # Predictions trace
                            fig.add_trace(
                                go.Scatter(
                                    x=recent_timestamps,
                                    y=recent_pred_data,
                                    mode='lines+markers',
                                    name=f'{var} Predicted',
                                    line=dict(color='red', width=2, dash='dash'),
                                    marker=dict(size=4),
                                    showlegend=(i == 0),  # Only show legend for first variable
                                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                                'Time: %{x}<br>' +
                                                'Value: %{y:.4f}<extra></extra>'
                                ),
                                row=row, col=col
                            )
                            
                            # Actuals trace
                            fig.add_trace(
                                go.Scatter(
                                    x=recent_timestamps,
                                    y=recent_actual_data,
                                    mode='lines+markers',
                                    name=f'{var} Actual',
                                    line=dict(color='blue', width=2),
                                    marker=dict(size=4),
                                    showlegend=(i == 0),  # Only show legend for first variable
                                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                                'Time: %{x}<br>' +
                                                'Value: %{y:.4f}<extra></extra>'
                                ),
                                row=row, col=col
                            )
                            
                            # Calculate and display error
                            if len(recent_pred_data) == len(recent_actual_data):
                                error = np.mean(np.abs(np.array(recent_pred_data) - np.array(recent_actual_data)))
                                self.stats['prediction_errors'][var] = error
            
            # Update layout
            fig.update_layout(
                height=600 * rows,  # Increased from 450 to 600 for even better visibility
                title_text="Real-Time Forecasting Dashboard",
                title_x=0.5,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update x-axis for all subplots with better time formatting
            fig.update_xaxes(
                title_text="Time", 
                showgrid=True,
                tickformat='%H:%M:%S',  # Format as HH:MM:SS
                tickangle=45  # Angle the time labels for better readability
            )
            fig.update_yaxes(title_text="Value", showgrid=True)
            
            return dcc.Graph(figure=fig, style={'height': f'{600 * rows}px'})
            
        except Exception as e:
            print(f"Error in _update_graphs: {e}")
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
    
    def add_buffer_predictions(self, predictions, actuals, current_step, current_datetime, variable_names):
        """
        Add new prediction data to the dashboard
        
        Args:
            predictions: List of prediction arrays for each time step
            actuals: List of actual arrays for each time step  
            current_step: Current step number
            current_datetime: Current timestamp
            variable_names: List of variable names
        """
        try:
            # Initialize variable names if first time
            if not self.variable_names:
                self.variable_names = variable_names
                for var in variable_names:
                    self.predictions_data[var] = deque(maxlen=self.max_points)
                    self.actuals_data[var] = deque(maxlen=self.max_points)
            
            # Add data for the first prediction step (t+1)
            if len(predictions) > 0 and len(actuals) > 0:
                pred_step = predictions[0]  # First prediction step
                actual_step = actuals[0]   # Corresponding actual
                
                # Add timestamp
                self.timestamps.append(current_datetime)
                
                # Add data for each variable
                for i, var in enumerate(variable_names):
                    if i < len(pred_step) and i < len(actual_step):
                        self.predictions_data[var].append(pred_step[i])
                        self.actuals_data[var].append(actual_step[i])
                
                # Update statistics
                self.current_step = current_step
                self.stats['total_predictions'] += 1
                self.stats['last_update'] = datetime.now().strftime("%H:%M:%S")
                
                print(f"Dashboard: Added data for step {current_step}, variables: {len(variable_names)}")
                
        except Exception as e:
            print(f"Error in add_buffer_predictions: {e}")
    
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
            if var in self.predictions_data:
                self.predictions_data[var].clear()
            if var in self.actuals_data:
                self.actuals_data[var].clear()
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
        predictions = [np.random.randn(len(variables)) * 10 + 50]
        actuals = [predictions[0] + np.random.randn(len(variables)) * 2]
        
        timestamp = datetime.now() + timedelta(seconds=step*5)
        
        plotter.add_buffer_predictions(
            predictions=predictions,
            actuals=actuals,
            current_step=step,
            current_datetime=timestamp,
            variable_names=variables
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
