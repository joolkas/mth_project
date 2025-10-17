#!/usr/bin/env python3
"""
Real-time Multi-step Forecasting with Zabbix Integration
Adapted from online_forecasting_multi_step.py for real system operation
"""

import json
import os
import sys
import time
import logging
import threading
import signal
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

# Add parent directory to import existing modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    from initial_model import get_initial_model, get_online_data
    from collect_data import ZabbixDataCollector
    from dash_plotter import DashRealTimePlotter
    from data_quality_handler import DataQualityHandler
    print("Imported existing modules including DataQualityHandler")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure parent modules are available")
    sys.exit(1)

# Import required modules from multi-step forecasting
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import matplotlib.pyplot as plt

# Ensure TensorFlow eager execution is enabled
tf.config.run_functions_eagerly(True)

# Global stop flag for graceful shutdown
stop_flag = threading.Event()

# Import helper functions from multi-step module
def predict_multistep_direct(model, context, variables, prediction_horizon=6):
    """Multi-step direct prediction - model outputs all future steps at once."""
    context_reshaped = context.reshape(1, context.shape[0], len(variables))
    multistep_pred = model.predict(context_reshaped, verbose=0)
    
    n_features = len(variables)
    predictions = []
    
    for step in range(prediction_horizon):
        start_idx = step * n_features
        end_idx = (step + 1) * n_features
        step_pred = multistep_pred[0, start_idx:end_idx]
        predictions.append(step_pred)
    
    return predictions

def inverse_difference(predictions_arrays, last_actual_values):
    """Convert differenced predictions back to actual values."""
    actual_predictions = []
    current_values = last_actual_values.copy()
    
    for pred_diff in predictions_arrays:
        current_values = current_values + np.array(pred_diff)
        actual_predictions.append(current_values.copy())
    
    return actual_predictions

def multistep_rolling_buffer_learning_prediction_with_dash_real(initial_model, 
                                        df_online, 
                                        scalers, 
                                        context_length,
                                        df_removed_nans_forecasting, 
                                        df_removed_nans_classification,
                                        dash_plotter, 
                                        variables, 
                                        prediction_horizon=6, 
                                        classification_model_path=None):
    """
    Multi-step forecasting adapted for real Zabbix data operation
    This is the key function that was modified to work with real data instead of artificial df_online
    """

    # Initialize graceful shutdown system
    global stop_flag
    stop_flag.clear()
    
    print("🚀 Starting real-time multi-step forecasting pipeline...")
    print(f"📊 Real data info: {len(df_online)} samples, {len(variables)} variables, context_length={context_length}")
    print(f"🎯 Variables: {variables}")
    print(f"📅 Time range: {df_online.index[0]} to {df_online.index[-1]}")
    
    # Prepare model for real-time operation
    model = initial_model
    
    # Apply differencing FIRST, then scaling (same as training)
    print("   Applying differencing to real Zabbix data (same as training)")
    df_online_differenced = df_online.diff().dropna()
    
    print("   Scaling differenced real data with training scalers")
    scaled_data = np.zeros_like(df_online_differenced.values)
    
    scaling_errors = []
    for i, var in enumerate(variables):
        if var in scalers:
            scaler = scalers[var]
            try:
                scaled_data[:, i] = scaler.transform(df_online_differenced[var].values.reshape(-1, 1)).flatten()
            except Exception as e:
                scaling_errors.append(f"{var}: {e}")
                # Fallback: use standardization
                data_values = df_online_differenced[var].values
                scaled_data[:, i] = (data_values - data_values.mean()) / (data_values.std() + 1e-8)
        else:
            scaling_errors.append(f"{var}: scaler not found")
            # Fallback: use standardization
            data_values = df_online_differenced[var].values
            scaled_data[:, i] = (data_values - data_values.mean()) / (data_values.std() + 1e-8)
    
    if scaling_errors:
        print(f"⚠️  Scaling issues: {len(scaling_errors)} variables had problems")
        for error in scaling_errors[:3]:
            print(f"   • {error}")
    
    # Real-time operation: Make ONE prediction using the most recent data
    print(f"   After differencing: {len(df_online)} → {len(df_online_differenced)} samples")
    print(f"   After scaling: {scaled_data.shape}")
    print(f"   Real-time mode: Making single prediction from most recent data")

    # Check if we have enough data for prediction
    if len(scaled_data) < context_length:
        print(f"⚠️  Insufficient data for prediction: need {context_length}, have {len(scaled_data)}")
        # Return empty DataFrames
        empty_df = pd.DataFrame(columns=variables)
        return empty_df, empty_df, empty_df, empty_df

    if dash_plotter is not None:
        try:
            dash_plotter.set_total_steps(1)  # Only one prediction step
        except Exception as e:
            print(f"⚠️  Dashboard configuration failed: {e}")
            dash_plotter = None

    # Use the most recent context_length samples for prediction
    current_context = scaled_data[-context_length:].copy()
    start_time = time.time()

    # Disable classification for production stability
    print("🔧 Classification disabled for production stability")
    classification_enabled = False

    # Single real-time prediction
    print("🎯 Making real-time multi-step prediction...")
    
    # Check for graceful shutdown
    if stop_flag.is_set():
        print("🛑 Graceful shutdown requested")
        empty_df = pd.DataFrame(columns=variables)
        return empty_df, empty_df, empty_df, empty_df

    # 1. Make multi-step predictions using direct method
    try:
        step_predictions = predict_multistep_direct(model, current_context, variables=variables, prediction_horizon=prediction_horizon)    
    except Exception as e:
        print(f"⚠️  Multi-step prediction failed ({e})")
        empty_df = pd.DataFrame(columns=variables)
        return empty_df, empty_df, empty_df, empty_df
                
    # 2. Convert all predictions to original scale
    step_predictions_original = []
    for pred in step_predictions:
        pred_original = []
        for i, var in enumerate(variables):
            # Filter status columns
            if 'status' in var.lower():
                pred[i] = np.round(pred[i])
            
            scaler = scalers[var]
            original_val = scaler.inverse_transform([[pred[i]]])[0, 0]
            pred_original.append(original_val)
        step_predictions_original.append(pred_original)
    
    # 3. For real-time operation, we don't have future actual values
    # Instead, create placeholder actuals for the prediction structure
    # In real-time, these would be filled as actual data arrives
    actuals_original = []
    for step in range(prediction_horizon):
        # Use last known values as placeholder actuals
        actual_original = []
        for i, var in enumerate(variables):
            # Get the last actual value from the original data
            last_actual_scaled = scaled_data[-1, i]
            scaler = scalers[var]
            last_actual_original = scaler.inverse_transform([[last_actual_scaled]])[0, 0]
            actual_original.append(last_actual_original)
        actuals_original.append(actual_original)
    
    # 4. Store predictions for output (only first prediction for immediate evaluation)
    final_predictions = []
    final_actuals = []
    final_timestamps = []
    predictions_actuals = []
    actuals_actuals = []
    
    if len(step_predictions_original) > 0:
        final_predictions.append(step_predictions_original[0])  # t+1 prediction
        final_actuals.append(actuals_original[0])  # placeholder actual
        # Use the last timestamp + 1 minute for the prediction timestamp
        final_timestamps.append(df_online_differenced.index[-1])

    # 5. Inverse differencing for real-time dashboard display
    # Use the last actual value from original data as base
    last_actual_values = df_online.iloc[-1][variables].values

    step_predictions_actual = inverse_difference(
        step_predictions_original, 
        last_actual_values
    )

    actuals_actual = inverse_difference(
        actuals_original, 
        last_actual_values
    )

    # 6. Real-time port status monitoring (simplified for production)
    port_statuses_check = df_removed_nans_classification.iloc[-1]  # Use most recent
    port_statuses = {}
    for name, status in port_statuses_check.items():
        if status not in [None, np.nan]:
            port_statuses[name] = status
    
    if len(step_predictions_actual) > 0 and len(actuals_actual) > 0:
        predictions_actuals.append(step_predictions_actual[0])
        actuals_actuals.append(actuals_actual[0])

    # 7. Send real-time data to Dash plotter
    if dash_plotter is not None:
        if len(step_predictions_actual) > 0:
            # Use the timestamp of the LAST ACTUAL DATA point, not current time
            # This ensures the dashboard shows actual data at the correct time
            current_timestamp = df_online.index[-1]  # Last actual data timestamp
            
            # Use the most recent actual values (last known real values) as "actuals"
            # Create actuals array that matches the dashboard expectation
            recent_actuals = []
            for step in range(prediction_horizon):
                # Use the last actual values from the original data
                recent_actuals.append(df_online.iloc[-1][variables].values.tolist())
            
            saved_prediction_t1 = step_predictions_actual[0] if len(step_predictions_actual) > 0 else None
            future_prediction_t1 = step_predictions_actual[0] if len(step_predictions_actual) > 0 else None
            
            try:
                dash_plotter.add_buffer_predictions(
                    predictions=step_predictions_actual, 
                    actuals=recent_actuals,  # Use most recent known actual values
                    current_step=0,  # Single step for real-time
                    current_datetime=current_timestamp,  # Current time when prediction is made
                    variable_names=variables,
                    saved_prediction=saved_prediction_t1,
                    future_prediction=future_prediction_t1,
                    port_statuses=port_statuses if len(port_statuses) > 0 else None,
                    classification_result=None  # Disabled for stability
                )
                print(f"✅ Dashboard updated with real-time data at {current_timestamp}")
            except Exception as e:
                print(f"Dashboard update failed: {e}")

    # Real-time prediction completed
    print("=" * 60)
    print("✅ Real-time multi-step prediction completed!")
    print(f"📈 Made single real-time prediction from Zabbix data")
    print(f"🎯 Generated {len(final_predictions)} predictions for immediate use")
    processing_time = time.time() - start_time
    print(f"⏱️  Total processing time: {processing_time:.3f} seconds")
    print(f"⚡ Efficient real-time operation: {processing_time:.3f} seconds per prediction")
    
    # Create results DataFrames
    try:
        if len(final_predictions) > 0:
            predictions_df = pd.DataFrame(
                data=final_predictions,
                columns=variables,
                index=pd.Index(range(context_length, context_length + len(final_predictions)), name='time_index')
            )
            
            actuals_df = pd.DataFrame(
                data=final_actuals,
                columns=variables,
                index=pd.Index(range(context_length, context_length + len(final_actuals)), name='time_index')
            )
            
            print(f"✅ Successfully created results for {len(final_predictions)} real-time predictions")
        else:
            predictions_df = pd.DataFrame(columns=variables)
            actuals_df = pd.DataFrame(columns=variables)
            print("⚠️  No predictions were completed")
        
        if len(predictions_actuals) > 0:
            predictions_actuals_df = pd.DataFrame(
                data=predictions_actuals,
                columns=variables,
                index=pd.Index(range(context_length, context_length + len(predictions_actuals)), name='time_index')
            )
            
            actuals_actuals_df = pd.DataFrame(
                data=actuals_actuals,
                columns=variables,
                index=pd.Index(range(context_length, context_length + len(actuals_actuals)), name='time_index')
            )
        else:
            predictions_actuals_df = pd.DataFrame(columns=variables)
            actuals_actuals_df = pd.DataFrame(columns=variables)
        
        print("✅ Real-time forecasting results prepared successfully!")
        
    except Exception as e:
        print(f"❌ Error creating results: {e}")
        empty_df = pd.DataFrame(columns=variables)
        predictions_df = actuals_df = predictions_actuals_df = actuals_actuals_df = empty_df

    return predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df


class ZabbixMultiStepForecastingLoop:
    """Real-time multi-step forecasting loop with Zabbix integration"""
    
    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        
        # Configuration
        self.context_length = self.config['model']['context_length']
        self.prediction_horizon = self.config['model']['prediction_horizon']
        self.model_path = self.config['model']['model_path']
        self.update_interval = self.config['data_collection']['update_interval']
        
        # Components
        self.model = None
        self.scalers = None
        self.variables = None
        self.collector = ZabbixDataCollector(config_file)
        self.monitoring_items = []
        self.dash_plotter = None
        self.quality_handler = None  # Data quality handler for Good/Bad flagging
        
        # Data storage
        os.makedirs('temp_data', exist_ok=True)
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration"""
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for clean shutdown"""
        def signal_handler(signum, frame):
            print(f"Received signal {signum}, initiating graceful shutdown...")
            stop_flag.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def setup_keyboard_listener(self):
        """Setup keyboard listener in separate thread"""
        def keyboard_listener():
            while not stop_flag.is_set():
                try:
                    user_input = input()
                    if user_input.upper() == 'Q':
                        print("Quit command detected! Initiating graceful shutdown...")
                        stop_flag.set()
                        break
                except (EOFError, KeyboardInterrupt):
                    break
                except Exception:
                    pass
        
        listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
        listener_thread.start()
        return listener_thread
    
    def initialize_system(self) -> bool:
        """Initialize all system components"""
        try:
            print("Initializing system components for real-time multi-step forecasting...")
            
            # Step 1: Connect to Zabbix
            print("Step 1: Connecting to Zabbix...")
            if not self.collector.connect():
                print("❌ Zabbix connection failed")
                return False
            print("✅ Zabbix connection successful")
            
            # Step 2: Check model directory
            print(f"Step 2: Checking model directory: {self.model_path}")
            if not os.path.exists(self.model_path):
                print(f"❌ Model directory not found: {self.model_path}")
                return False
            print("✅ Model directory exists")

            # Step 3: Load trained model
            print("Step 3: Loading trained multi-step model...")
            try:
                self.model = get_initial_model(self.model_path)
                print("✅ Multi-step model loaded successfully")
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
                return False
            
            # Step 4: Load scalers and variables
            print("Step 4: Loading scalers and variables...")
            try:
                df_online, self.scalers, context_length, df_removed_nans_forecasting, df_removed_nans_classification, self.variables, model_mode = get_online_data(self.model_path)
                print("✅ Scalers and variables loaded")
            except Exception as e:
                print(f"❌ Scalers loading failed: {e}")
                return False
            
            # Step 5: Verify parameters
            if context_length != self.context_length:
                print(f"⚠️ Context length mismatch: config={self.context_length}, model={context_length}")
                self.context_length = context_length
            
            # Step 6: Discover monitoring items
            print("Step 6: Discovering Zabbix monitoring items...")
            try:
                self.monitoring_items = self.collector.discover_items()
                if not self.monitoring_items:
                    print("❌ No monitoring items found")
                    return False
            except Exception as e:
                print(f"❌ Item discovery failed: {e}")
                return False

            # Step 7: Initialize Data Quality Handler  
            print("Step 7: Initializing Data Quality Handler for Good/Bad flagging...")
            try:
                # Configure cache age based on update interval (2x update interval)
                cache_age_minutes = max(10, self.update_interval * 2)  # At least 10 minutes
                self.quality_handler = DataQualityHandler(
                    variables=self.variables,
                    max_cache_age_minutes=cache_age_minutes
                )
                print(f"✅ Data Quality Handler initialized with {cache_age_minutes}min cache age")
                print("   Features: Good/Bad sample flagging + last-value interpolation")
            except Exception as e:
                print(f"❌ Data Quality Handler initialization failed: {e}")
                return False

            print("Multi-step forecasting system initialization completed!")
            print(f"   Model mode: {model_mode}")
            print(f"   Context length: {context_length}")
            print(f"   Prediction horizon: {self.prediction_horizon}")
            print(f"   Variables: {len(self.variables)}")
            print(f"   Monitoring items: {len(self.monitoring_items)}")
            print(f"   Quality-based data handling: ✅ Enabled (Good/Bad flagging + interpolation)")

            return True
            
        except Exception as e:
            print(f"❌ Unexpected error during system initialization: {e}")
            return False
    
    def initialize_dashboard(self):
        """Initialize Dash dashboard for real-time visualization"""
        try:
            dashboard_port = self.config['monitoring']['dashboard_port']
            print(f"🌐 Initializing real-time dashboard on port {dashboard_port}...")
            
            dashboard_update_interval = self.update_interval * 1000
            self.dash_plotter = DashRealTimePlotter(update_interval=dashboard_update_interval)
            
            if hasattr(self.dash_plotter, 'start_server'):
                print("📊 Starting dashboard server...")
                dashboard_thread = self.dash_plotter.start_server(
                    host='0.0.0.0',
                    port=dashboard_port,
                    debug=False
                )
                print("✅ Dashboard started")
                time.sleep(3)
            else:
                def start_dashboard():
                    try:
                        self.dash_plotter.app.run_server(
                            host='0.0.0.0',
                            port=dashboard_port,
                            debug=False,
                            threaded=True,
                            use_reloader=False
                        )
                    except Exception as e:
                        print(f"Dashboard server failed: {e}")
                
                dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
                dashboard_thread.start()
                print("✅ Dashboard started in thread mode")
                time.sleep(3)
            
            # Display connection info
            import socket
            hostname = socket.gethostname()
            try:
                local_ip = socket.gethostbyname(hostname)
            except:
                local_ip = "localhost"
            
            print(f"🌐 Dashboard accessible at: http://{local_ip}:{dashboard_port}")
            
        except Exception as e:
            print(f"⚠️ Dashboard initialization failed: {e}")
            print("   Continuing without dashboard...")
            self.dash_plotter = None
    
    def collect_and_prepare_data(self) -> Optional[pd.DataFrame]:
        """Collect current data from Zabbix and prepare for prediction using Data Quality Handler"""
        try:
            # Step 1: Collect RAW data from Zabbix 
            print("📡 Collecting RAW data from Zabbix...")
            raw_data = self._collect_raw_zabbix_data(hours_back=1)

            # Match columns to model variables
            if raw_data is not None and not raw_data.empty:
                print(f"📊 Raw Zabbix data collected: {raw_data.shape}")
                print(f"📅 Data time range: {raw_data.index[0]} to {raw_data.index[-1]}")
                
                available_columns = raw_data.columns.tolist()
                print(f"🏷️  Available columns: {len(available_columns)}, Model variables: {len(self.variables)}")
                
                # Map available columns to model variables
                if len(available_columns) >= len(self.variables):
                    selected_data = raw_data[available_columns[:len(self.variables)]].copy()
                    selected_data.columns = self.variables
                else:
                    selected_data = pd.DataFrame(index=raw_data.index, columns=self.variables)
                    for i, col in enumerate(available_columns):
                        if i < len(self.variables):
                            selected_data[self.variables[i]] = raw_data[col]
                
                # Remove infinite values but keep zeros for quality assessment
                selected_data = selected_data.replace([np.inf, -np.inf], np.nan)
                
                # Show sample of raw data BEFORE quality processing
                if len(selected_data) > 0:
                    print("🔍 Sample RAW data (last 3 rows - before quality processing):")
                    for i in range(max(0, len(selected_data)-3), len(selected_data)):
                        timestamp = selected_data.index[i]
                        print(f"   {timestamp}: ", end="")
                        for col in selected_data.columns[:3]:  # Show first 3 columns
                            value = selected_data[col].iloc[i]
                            if pd.isna(value):
                                print(f"{col}=Missing ", end="")
                            else:
                                print(f"{col}={value:.3f} ", end="")
                        print()
                            
            else:
                print("⚠️  No fresh data from Zabbix - will use cached values only")
                selected_data = pd.DataFrame()

            # Step 2: Apply Data Quality Handler with Good/Bad flagging
            print("🏷️  Processing data with Quality Handler (Good/Bad flagging + interpolation)...")
            
            # Request sufficient data points for model context + small buffer
            target_length = self.context_length + 10
            
            # Process data with quality flags
            processed_data, quality_flags = self.quality_handler.process_data_with_quality_flags(
                raw_data=selected_data,
                target_length=target_length,
                target_frequency='1min'  # 1-minute intervals
            )
            
            # Show Quality Handler status
            quality_summary = self.quality_handler.get_quality_summary()
            print(f"📊 Data Quality Summary:")
            print(f"   Total requests: {quality_summary['total_requests']}")
            print(f"   Variables with cache: {quality_summary['cached_variables']}/{len(self.variables)}")
            
            # Show processed data with quality flags
            if len(processed_data) > 0:
                print("🏷️  Quality-processed data (last 3 rows with Good/Bad flags):")
                for i in range(max(0, len(processed_data)-3), len(processed_data)):
                    timestamp = processed_data.index[i]
                    print(f"   {timestamp}: ", end="")
                    for col in processed_data.columns[:3]:  # Show first 3 columns
                        value = processed_data[col].iloc[i]
                        quality = quality_flags[col].iloc[i]
                        print(f"{col}={value:.3f}({quality}) ", end="")
                    print()
                
                # Quality statistics for network variables
                network_cols = [col for col in processed_data.columns if 'bits' in col.lower() or 'network' in col.lower()]
                for col in network_cols[:2]:  # Check first 2 network columns
                    if col in quality_summary['variable_stats']:
                        var_stats = quality_summary['variable_stats'][col]
                        good_pct = var_stats['good_percentage']
                        print(f"   📊 {col}: {good_pct:.1f}% Good data, {var_stats['interpolated_count']} interpolated")
                        
                        # Show recent quality pattern
                        recent_flags = quality_flags[col].tail(5).tolist()
                        print(f"      Recent quality: {recent_flags}")
            
            # Final validation
            if len(processed_data) < self.context_length:
                print(f"⚠️  Insufficient data after quality processing: need {self.context_length}, have {len(processed_data)}")
                return None
            
            print(f"✅ Quality-based data preparation completed: {processed_data.shape}")
            print("   Data now ready for prediction model (no more artificial zeros!)")
            return processed_data
            
        except Exception as e:
            print(f"❌ Quality-based data preparation failed: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _collect_raw_zabbix_data(self, hours_back: int = 1) -> Optional[pd.DataFrame]:
        """
        Collect raw data from Zabbix without built-in gap filling
        This bypasses the collect_recent_data method that includes gap filling
        """
        try:
            time_to = int(time.time())
            time_from = int(time_to - (hours_back * 3600))
            
            print(f"🌐 Collecting RAW Zabbix data (bypassing gap filling)...")
            print(f"   Time range: {datetime.fromtimestamp(time_from)} to {datetime.fromtimestamp(time_to)}")
            
            # Get raw history data directly from Zabbix API
            item_ids = [item['itemid'] for item in self.monitoring_items]
            all_data = []
            
            # Separate by value type as in the original collector
            float_items = [item for item in self.monitoring_items if item.get('value_type') == '0']
            uint_items = [item for item in self.monitoring_items if item.get('value_type') == '3']
            
            history = []
            
            # Get history data for both types
            if float_items:
                float_item_ids = [item['itemid'] for item in float_items]
                float_history = self.collector.zabbix_api.history.get(
                    itemids=float_item_ids,
                    time_from=time_from,
                    time_till=time_to,
                    output='extend',
                    sortfield='clock',
                    history=0  # Float history
                )
                history.extend(float_history)
            
            if uint_items:
                uint_item_ids = [item['itemid'] for item in uint_items]
                uint_history = self.collector.zabbix_api.history.get(
                    itemids=uint_item_ids,
                    time_from=time_from,
                    time_till=time_to,
                    output='extend',
                    sortfield='clock',
                    history=3  # Integer history
                )
                history.extend(uint_history)
            
            # Process raw data without gap filling
            item_lookup = {item['itemid']: item for item in self.monitoring_items}
            
            for record in history:
                if record['itemid'] in item_lookup:
                    try:
                        # Apply same UTC offset as in collect_data.py for consistency
                        timestamp = pd.to_datetime(int(record['clock']) + (self.collector.server_utc_offset * 3600), unit='s')
                        value = float(record['value'])
                        variable_name = item_lookup[record['itemid']]['display_name']
                        
                        all_data.append({
                            'timestamp': timestamp,
                            'variable': variable_name,
                            'value': value
                        })
                    except (ValueError, TypeError):
                        continue
            
            if not all_data:
                print("   ❌ No raw data collected")
                return None
            
            # Create simple pivot table without resampling or gap filling
            df = pd.DataFrame(all_data)
            df_pivot = df.pivot_table(
                index='timestamp',
                columns='variable',
                values='value',
                aggfunc='last'
            )
            
            print(f"   📊 Raw data collected: {df_pivot.shape}")
            print(f"   🔍 Raw data contains actual zeros from Zabbix (no artificial filling yet)")
            
            return df_pivot
            
        except Exception as e:
            print(f"❌ Raw Zabbix data collection failed: {e}")
            return None
    


    def run_prediction_cycle(self, df: pd.DataFrame, cycle: int) -> bool:
        """Run single prediction cycle using real Zabbix data"""
        try:
            # Save temporary data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = f"temp_data/multistep_cycle_{cycle:04d}_{timestamp}.csv"
            df.to_csv(temp_file)
            
            print("🏷️  Quality-processed Zabbix data going to multi-step model:")
            print(f"Data shape: {df.shape}")
            
            # Show current data values
            print(f"📈 Current data values (timestamp: {df.index[-1]}):")
            for col in df.columns:
                current_value = df[col].iloc[-1]
                print(f"   {col}: {current_value:.3f}")
            
            # Show data quality statistics
            quality_summary = self.quality_handler.get_quality_summary()
            print(f"📊 Data Quality Summary for Cycle {cycle}:")
            
            total_good = sum(stats['good_count'] for stats in quality_summary['variable_stats'].values())
            total_interpolated = sum(stats['interpolated_count'] for stats in quality_summary['variable_stats'].values())
            total_samples = total_good + total_interpolated
            
            if total_samples > 0:
                good_percentage = (total_good / total_samples) * 100
                print(f"   Overall: {good_percentage:.1f}% Good data, {total_interpolated} interpolated samples")
                print(f"   Quality Handler successfully handled missing/bad values! ✅")

            # Don't clear dashboard data - we want to maintain history of 60 points
            # if self.dash_plotter is not None:
            #     self.dash_plotter.clear_data()
            
            # Run multi-step forecasting with real Zabbix data
            predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df = multistep_rolling_buffer_learning_prediction_with_dash_real(
                initial_model=self.model,
                df_online=df,  # This is now real Zabbix data instead of artificial df_online
                scalers=self.scalers,
                context_length=self.context_length,
                df_removed_nans_forecasting=df,
                df_removed_nans_classification=df,
                dash_plotter=self.dash_plotter,
                variables=self.variables,
                prediction_horizon=self.prediction_horizon
            )
            
            # Log real-time results
            if not predictions_df.empty:
                # Note: In real-time operation, actuals_df contains placeholder values (last known actuals)
                # so MAE and percentage error are not meaningful accuracy measures
                mae = np.mean(np.abs(predictions_df.values - actuals_df.values)) if not actuals_df.empty else 0
                
                # Display prediction summary without misleading accuracy metrics
                print(f"Multi-step Cycle {cycle}: Generated {len(predictions_df)} predictions")
                print(f"   Prediction range: {np.min(predictions_df.values):.3f} to {np.max(predictions_df.values):.3f}")
                print(f"   Note: Real-time mode - accuracy metrics require future actual values")
            else:
                print(f"Multi-step Cycle {cycle}: No predictions generated")

            return True
            
        except Exception as e:
            print(f"Multi-step prediction cycle {cycle} failed: {e}")
            return False
    
    def run_monitoring_loop(self):
        """Main real-time monitoring loop"""
        print("🏭 Starting real-time multi-step forecasting loop with Zabbix integration...")
        print("Press 'Q' + Enter to stop gracefully")

        # Setup signal handlers and keyboard listener
        self.setup_signal_handlers()
        keyboard_thread = self.setup_keyboard_listener()
        
        cycle = 0
        start_time = time.time()
        
        try:
            while not stop_flag.is_set():
                cycle += 1
                cycle_start = time.time()
                
                print(f"🔄 Multi-step Cycle #{cycle} started")
                
                # Collect and prepare real data from Zabbix
                prepared_data = self.collect_and_prepare_data()
                
                if prepared_data is None:
                    print(f"Multi-step Cycle #{cycle}: Skipping due to data issues")
                else:
                    # Run multi-step prediction with real data
                    success = self.run_prediction_cycle(prepared_data, cycle)
                    
                    if success:
                        print(f"✅ Multi-step Cycle #{cycle} completed")
                        print("-" * 40)
                    else:
                        print(f"⚠️ Multi-step Cycle #{cycle} had issues")

                # Wait for next cycle (real-time operation every minute)
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.update_interval - cycle_time)
                
                # Interruptible sleep
                end_time = time.time() + sleep_time
                while time.time() < end_time and not stop_flag.is_set():
                    time.sleep(0.1)
                
        except Exception as e:
            print(f"Unexpected error in monitoring loop: {e}")

        finally:
            # Clean shutdown
            total_time = time.time() - start_time
            print("🛑 Graceful shutdown initiated")
            print(f"📊 Multi-step forecasting session summary:")
            print(f"   Total cycles: {cycle}")
            print(f"   Total runtime: {total_time:.1f} seconds")
            print(f"   Average cycle time: {total_time/max(cycle,1):.1f} seconds")
            print("✅ Multi-step forecasting shutdown completed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Zabbix Real-time Multi-step Forecasting Loop')
    parser.add_argument('--test', action='store_true', help='Test system components')
    args = parser.parse_args()
    
    forecaster = ZabbixMultiStepForecastingLoop()
    
    if args.test:
        print("🧪 Testing multi-step forecasting system components...")
        print("=" * 50)
        
        try:
            success = forecaster.initialize_system()
            if success:
                print("=" * 50)
                print("✅ All multi-step forecasting systems ready!")
                print(f"   Found {len(forecaster.monitoring_items)} monitoring items")
                print(f"   Model variables: {len(forecaster.variables)}")
                print(f"   Context length: {forecaster.context_length}")
                print(f"   Prediction horizon: {forecaster.prediction_horizon}")
            else:
                print("=" * 50)
                print("❌ Multi-step system initialization failed")
                return 1
        except Exception as e:
            print(f"❌ Unexpected error during testing: {e}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            return 1
        
        return 0
    
    # Initialize and run multi-step monitoring
    print("🏭 Zabbix Real-time Multi-step Anomaly Detection")
    print("=" * 40)
    print("Initializing multi-step forecasting system...")
    
    if not forecaster.initialize_system():
        print("❌ Multi-step system initialization failed")
        return 1
    
    print("✅ Multi-step system ready!")
    
    # Initialize dashboard
    forecaster.initialize_dashboard()

    print("Starting multi-step monitoring loop...")
    print("Press 'Q' + Enter to stop gracefully")
    print("=" * 40)
    
    try:
        forecaster.run_monitoring_loop()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    return 0


if __name__ == "__main__":
    exit(main())