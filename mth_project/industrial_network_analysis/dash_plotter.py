import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import threading
import time

class DashRealTimePlotter:
    """
    Lightweight multi-stream real-time plotter optimized for many variables.
    Minimal implementation with only essential methods used by test_online_approach_3.py
    """

    def __init__(self, update_interval=1000, prediction_horizon=3, port=8050, max_points=200):
        """
        Initialize lightweight multi-stream plotter.
        
        Args:
            update_interval: Update frequency in milliseconds
            prediction_horizon: Number of prediction steps  
            port: Server port
            max_points: Maximum points to keep in memory (for performance)
        """
        self.update_interval = update_interval
        self.prediction_horizon = prediction_horizon
        self.port = port
        self.max_points = max_points
        
        # Thread-safe data storage with automatic size limiting
        self.data_lock = threading.Lock()
        self.streams = {}  # Will be populated when first data arrives
        self.timestamps = deque(maxlen=max_points)
        
        # Status tracking
        self.current_step = 0
        self.total_steps = 0
        
        # Color palette for streams (optimized for many variables)
        self.colors = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#34495e', '#95a5a6', '#d35400',
            '#c0392b', '#8e44ad', '#2980b9', '#27ae60', '#f1c40f'
        ]
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Layout for 3-trend multi-stream visualization: Actuals, Saved Preds, Forward Preds."""
        self.app.layout = html.Div([
            html.H1("📊 Multi-Stream Buffer Analysis", 
                   style={'textAlign': 'center', 'marginBottom': 20, 'color': '#2c3e50'}),
            
            # Compact info panel
            html.Div([
                html.Div(id='progress-info', style={'display': 'inline-block', 'margin': 10}),
                html.Div(id='stream-stats', style={'display': 'inline-block', 'margin': 10}),
            ], style={'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'padding': 10}),
            
            # Main plot
            dcc.Graph(id='multi-stream-plot', style={'height': '75vh'}),
            
            # Auto-refresh
            dcc.Interval(id='interval-component', interval=self.update_interval, n_intervals=0),
        ], style={'padding': 10, 'fontFamily': 'Arial, sans-serif'})

    def _setup_callbacks(self):
        """Minimal callbacks for multi-stream updates."""
        @self.app.callback(
            [Output('multi-stream-plot', 'figure'),
             Output('progress-info', 'children'),
             Output('stream-stats', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_all(n_intervals):
            return self._create_figure(), self._get_progress(), self._get_stats()

    def _create_figure(self):
        """Create multi-stream subplot figure with 3 trends per stream: actuals, forward preds, saved preds."""
        with self.data_lock:
            if not self.streams:
                # Empty state
                fig = go.Figure()
                fig.update_layout(
                    title="⏳ Waiting for multi-stream data...",
                    annotations=[dict(
                        text="📈 Multi-stream predictions will appear here<br>🔴 Actuals | 🔵 Forward Preds | 🟢 Saved Preds",
                        x=0.5, y=0.5, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=16)
                    )]
                )
                return fig
            
            # Determine subplot layout
            n_streams = len(self.streams)
            if n_streams <= 2:
                rows, cols = 1, n_streams
            elif n_streams <= 4:
                rows, cols = 2, 2  
            elif n_streams <= 6:
                rows, cols = 2, 3
            elif n_streams <= 9:
                rows, cols = 3, 3
            else:
                rows, cols = 4, max(4, (n_streams + 3) // 4)
            
            # Create subplots
            stream_names = list(self.streams.keys())
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=stream_names[:rows*cols],
                vertical_spacing=0.08,
                horizontal_spacing=0.03
            )
            
            # Add traces for each stream (3 lines per stream)
            for idx, stream_name in enumerate(stream_names[:rows*cols]):
                row = (idx // cols) + 1
                col = (idx % cols) + 1
                
                stream_data = self.streams[stream_name]
                timestamps = list(self.timestamps)
                
                # 1. Actual values (red solid line)
                if stream_data['actual']:
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=list(stream_data['actual']),
                            mode='lines',
                            name='Actual',
                            line=dict(color='#e74c3c', width=2),  # Red for actual
                            showlegend=(idx == 0)
                        ),
                        row=row, col=col
                    )
                
                # 2. Saved predictions (green dashed line - historical kept predictions)
                if stream_data['kept']:
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=list(stream_data['kept']),
                            mode='lines',
                            name='Saved Predictions',
                            line=dict(color='#27ae60', width=2, dash='dash'),  # Green dashed
                            showlegend=(idx == 0)
                        ),
                        row=row, col=col
                    )
                
                # 3. Forward predictions (blue dotted line - future trajectory)
                if stream_data['kept'] and len(timestamps) > 0:
                    # Create forward prediction line by extending from last saved prediction
                    forward_times = []
                    forward_values = []
                    
                    # Start from current time
                    current_time = timestamps[-1]
                    last_saved = list(stream_data['kept'])[-1] if stream_data['kept'] else 0
                    
                    # Add current point as connection
                    forward_times.append(current_time)
                    forward_values.append(last_saved)
                    
                    # Add prediction points for each horizon
                    for h in range(1, self.prediction_horizon + 1):
                        # Calculate future time
                        if isinstance(current_time, datetime):
                            future_time = current_time + timedelta(minutes=h)
                        else:
                            future_time = current_time + h
                        
                        # Get prediction value for this horizon
                        if h == 1 and stream_data['kept']:
                            pred_value = list(stream_data['kept'])[-1]  # Use last saved as t+1
                        else:
                            h_key = f't+{h}'
                            if h_key in stream_data and stream_data[h_key]:
                                pred_value = list(stream_data[h_key])[-1]
                            else:
                                pred_value = forward_values[-1]  # Use last available prediction
                        
                        forward_times.append(future_time)
                        forward_values.append(pred_value)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=forward_times,
                            y=forward_values,
                            mode='lines+markers',
                            name=f'Forward Predictions (t+1 to t+{self.prediction_horizon})',
                            line=dict(color='#3498db', width=2, dash='dot'),  # Blue dotted
                            marker=dict(size=4),
                            showlegend=(idx == 0)
                        ),
                        row=row, col=col
                    )
            
            # Compact layout
            fig.update_layout(
                title=f"📊 Multi-Stream Analysis - Step {self.current_step} (3 Trends per Stream)",
                height=600,
                showlegend=True,
                legend=dict(orientation="h", y=1.02, x=0.5, xanchor='center'),
                margin=dict(l=40, r=40, t=80, b=40)
            )
            
            return fig

    def _get_progress(self):
        """Get progress info."""
        if self.total_steps > 0:
            pct = (self.current_step / self.total_steps) * 100
            return f"📊 Step {self.current_step}/{self.total_steps} ({pct:.1f}%)"
        return f"📊 Step {self.current_step}"

    def _get_stats(self):
        """Get stream statistics."""
        with self.data_lock:
            n_streams = len(self.streams)
            total_points = len(self.timestamps)
            return f"🔢 {n_streams} streams • {total_points} points • Actuals + Saved + Forward (t+1 to t+{self.prediction_horizon})"

    def add_buffer_predictions(self, predictions, actuals, current_step, current_datetime=None, variable_names=None):
        """
        Add multi-stream buffer predictions - optimized for many streams.
        
        Args:
            predictions: List of prediction arrays [t+1, t+2, t+3, ...] 
                        Each array shape: (n_variables,)
            actuals: List of actual arrays [t+1, t+2, t+3, ...]
                    Each array shape: (n_variables,)  
            current_step: Current step index
            current_datetime: Current timestamp
            variable_names: List of variable names for stream labeling
        """
        if not predictions or not actuals:
            return
            
        with self.data_lock:
            # Initialize streams on first call
            if not self.streams:
                n_vars = len(predictions[0]) if len(predictions) > 0 else 0
                for i in range(n_vars):
                    if variable_names and i < len(variable_names):
                        stream_name = str(variable_names[i])
                    else:
                        stream_name = f"Var_{i+1}"
                    
                    self.streams[stream_name] = {
                        'actual': deque(maxlen=self.max_points),
                        'kept': deque(maxlen=self.max_points),  # t+1
                        **{f't+{h}': deque(maxlen=self.max_points) 
                           for h in range(2, self.prediction_horizon + 1)}
                    }
            
            # Add timestamp
            timestamp = current_datetime if current_datetime is not None else current_step
            self.timestamps.append(timestamp)
            
            # Add data for each stream/variable
            stream_names = list(self.streams.keys())
            for var_idx, stream_name in enumerate(stream_names):
                stream = self.streams[stream_name]
                
                # Add actual value (from t+1 actuals, representing current actual)
                if len(actuals) > 0 and len(actuals[0]) > var_idx:
                    stream['actual'].append(actuals[0][var_idx])
                else:
                    stream['actual'].append(None)
                
                # Add predictions for each horizon
                for horizon_idx, pred_array in enumerate(predictions):
                    h = horizon_idx + 1
                    if h == 1:
                        # Kept predictions (t+1)
                        if len(pred_array) > var_idx:
                            stream['kept'].append(pred_array[var_idx])
                        else:
                            stream['kept'].append(None)
                    else:
                        # Buffer predictions (t+2, t+3, ...)
                        h_key = f't+{h}'
                        if h_key in stream:
                            if len(pred_array) > var_idx:
                                stream[h_key].append(pred_array[var_idx])
                            else:
                                stream[h_key].append(None)
            
            self.current_step = current_step

    def set_total_steps(self, total_steps):
        """Set total steps for progress tracking."""
        self.total_steps = total_steps

    def start_server(self, debug=False, threaded=True):
        """Start Dash server in background thread."""
        def run():
            print(f"🚀 Multi-stream plotter: http://localhost:{self.port}")
            self.app.run(debug=debug, port=self.port, host='127.0.0.1')
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
        return thread

    def get_statistics(self):
        """Get comprehensive statistics."""
        with self.data_lock:
            total_points = len(self.timestamps)
            n_streams = len(self.streams)
            
            return {
                'total_kept_predictions': total_points * n_streams,
                'total_buffer_predictions': total_points * n_streams * (self.prediction_horizon - 1),
                'n_streams': n_streams,
                'prediction_horizon': self.prediction_horizon,
                'current_step': self.current_step,
                'total_steps': self.total_steps,
                'progress_percentage': (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0,
                'max_points': self.max_points
            }