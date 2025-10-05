#!/usr/bin/env python3
"""
🏭 Main Forecasting Loop for Simplified Zabbix Integration

Real-time anomaly detection using LSTM forecasting with 1-minute cycles.
Downloads data from Zabbix, temporarily stores it, and runs predictions.

Author: Industrial Network Analysis System
Version: 1.0
"""

import sys
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pickle
from pyzabbix import ZabbixAPI
import urllib3

# Import TensorFlow configuration fix
from tensorflow_config import fix_tensorflow_configuration

# Disable SSL warnings
urllib3.disable_warnings()

# Fix TensorFlow configuration early
fix_tensorflow_configuration()

# Add parent directory to import existing modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    # Mock the get_data module to prevent import errors
    import types
    mock_get_data = types.ModuleType('get_data')
    mock_get_data.get_processed_path = lambda: ("", "")
    sys.modules['get_data'] = mock_get_data
    
    from initial_model import get_initial_model, get_online_data
    from online_forecasting_multi_step import multistep_rolling_buffer_learning_prediction_with_dash
    
    # Use standalone dashboard instead of parent directory version
    from standalone_dashboard import StandaloneDashboard
    print("✅ Successfully imported modules with standalone dashboard")
except ImportError as e:
    print(f"❌ Cannot import required modules: {e}")
    print("   This indicates the parent directory modules are not available")
    print("   Please ensure you're running from the correct location")
    print("   Or install this as a standalone system")
    sys.exit(1)


class ZabbixForecastingLoop:
    """Main forecasting loop with Zabbix integration"""
    
    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # Connection components
        self.zabbix_api = None
        self.model = None
        self.scalers = None
        self.dash_plotter = None
        
        # System parameters
        self.context_length = self.config['model']['context_length']
        self.prediction_horizon = self.config['model']['prediction_horizon']
        self.update_interval = self.config['data_collection']['update_interval']
        self.model_path = self.config['model']['model_path']
        
        # Search criteria for data collection
        self.search_criteria = self.config['data_collection']['search_criteria']
        self.host_groups = self.config['data_collection']['host_groups']
        
        # Temporary data storage for rolling window
        self.data_buffer = None
        self.variables = None
        self.item_cache = {}
        self.host_cache = {}
        
        # Initialize data storage directory
        os.makedirs('temp_data', exist_ok=True)
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Config error: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """Setup logging"""
        log_level = getattr(logging, self.config['monitoring']['log_level'], logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('forecasting_loop.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def connect_zabbix(self) -> bool:
        """Connect to Zabbix API"""
        try:
            zabbix_config = self.config['zabbix']
            self.logger.info(f"🔗 Connecting to Zabbix: {zabbix_config['url']}")
            
            self.zabbix_api = ZabbixAPI(zabbix_config['url'])
            self.zabbix_api.session.verify = False
            self.zabbix_api.login(zabbix_config['user'], zabbix_config['password'])
            
            version = self.zabbix_api.apiinfo.version()
            self.logger.info(f"✅ Connected to Zabbix {version}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Zabbix connection failed: {e}")
            return False
    
    def load_trained_model(self) -> bool:
        """Load the trained model and associated data"""
        try:
            self.logger.info(f"🤖 Loading trained model from: {self.model_path}")
            
            # Load the model using your existing function
            self.model = get_initial_model(self.model_path)
            
            # Load online data and scalers
            df_online, scalers, context_length, df_removed_nans_forecasting, df_removed_nans_classification, variables, model_mode = get_online_data(self.model_path)
            
            self.scalers = scalers
            self.variables = variables
            self.df_removed_nans_forecasting = df_removed_nans_forecasting
            self.df_removed_nans_classification = df_removed_nans_classification
            
            # Verify parameters match configuration
            if context_length != self.context_length:
                self.logger.warning(f"⚠️ Context length mismatch: config={self.context_length}, model={context_length}")
                self.context_length = context_length
            
            self.logger.info(f"✅ Model loaded successfully")
            self.logger.info(f"   Variables: {len(variables)}")
            self.logger.info(f"   Context length: {context_length}")
            self.logger.info(f"   Model mode: {model_mode}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def discover_monitoring_items(self) -> List[Dict]:
        """Discover items to monitor based on search criteria"""
        try:
            if self.item_cache:
                return self.item_cache
            
            # Get hosts from configured groups
            all_hosts = []
            for group_name in self.host_groups:
                try:
                    groups = self.zabbix_api.hostgroup.get(filter={"name": group_name})
                    if groups:
                        hosts = self.zabbix_api.host.get(
                            groupids=[groups[0]['groupid']],
                            output=['hostid', 'host', 'name'],
                            filter={'status': 0}
                        )
                        all_hosts.extend(hosts)
                except Exception as e:
                    self.logger.warning(f"⚠️ Error accessing group '{group_name}': {e}")
            
            if not all_hosts:
                self.logger.error("❌ No hosts found")
                return []
            
            # Get monitored items
            host_ids = [host['hostid'] for host in all_hosts]
            items = self.zabbix_api.item.get(
                hostids=host_ids,
                output=['itemid', 'name', 'key_', 'hostid'],
                monitored=True,
                filter={'value_type': [0, 3]}  # Numeric values only
            )
            
            # Filter items based on search criteria and match to model variables
            host_lookup = {host['hostid']: host for host in all_hosts}
            filtered_items = []
            
            for item in items:
                item_name_lower = item['name'].lower()
                item_key_lower = item['key_'].lower()
                
                # Check if any search criteria matches
                for criteria in self.search_criteria:
                    if (criteria.lower() in item_name_lower or 
                        criteria.lower() in item_key_lower):
                        
                        if item['hostid'] in host_lookup:
                            item['host_name'] = host_lookup[item['hostid']]['host']
                            item['display_name'] = f"{item['host_name']}_{item['name']}"
                            filtered_items.append(item)
                        break
            
            self.item_cache = filtered_items
            self.logger.info(f"🔍 Discovered {len(filtered_items)} monitoring items")
            
            return filtered_items
            
        except Exception as e:
            self.logger.error(f"❌ Item discovery failed: {e}")
            return []
    
    def collect_current_data(self, items: List[Dict]) -> Optional[pd.DataFrame]:
        """Collect current data from Zabbix for the monitoring cycle"""
        try:
            # Get data from the last 2 hours to ensure we have enough context
            time_to = int(time.time())
            time_from = time_to - (2 * 3600)  # 2 hours back
            
            all_data = []
            item_ids = [item['itemid'] for item in items]
            
            # Collect history data
            history = self.zabbix_api.history.get(
                itemids=item_ids,
                time_from=time_from,
                time_till=time_to,
                output='extend',
                sortfield='clock'
            )
            
            # Process data
            item_lookup = {item['itemid']: item for item in items}
            
            for record in history:
                if record['itemid'] in item_lookup:
                    try:
                        all_data.append({
                            'timestamp': pd.to_datetime(int(record['clock']), unit='s'),
                            'variable': item_lookup[record['itemid']]['display_name'],
                            'value': float(record['value'])
                        })
                    except (ValueError, TypeError):
                        continue
            
            if not all_data:
                self.logger.warning("⚠️ No current data collected")
                return None
            
            # Create time series DataFrame
            df = pd.DataFrame(all_data)
            df_pivot = df.pivot_table(
                index='timestamp',
                columns='variable',
                values='value',
                aggfunc='mean'
            )
            
            # Clean and resample data
            df_pivot = df_pivot.sort_index()
            df_pivot = df_pivot.fillna(method='ffill', limit=5)
            df_pivot = df_pivot.fillna(method='bfill', limit=5)
            df_pivot = df_pivot.fillna(0)
            
            # Resample to 1-minute intervals
            df_resampled = df_pivot.resample('1T').mean()
            df_resampled = df_resampled.fillna(method='ffill')
            
            return df_resampled
            
        except Exception as e:
            self.logger.error(f"❌ Current data collection failed: {e}")
            return None
    
    def prepare_data_for_prediction(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare data for prediction by matching to model variables"""
        try:
            if df is None or df.empty:
                return None
            
            # Ensure we have enough data for context
            if len(df) < self.context_length:
                self.logger.warning(f"⚠️ Insufficient data: need {self.context_length}, have {len(df)}")
                return None
            
            # Try to match columns to model variables
            # This is simplified - in practice you might need more sophisticated matching
            available_columns = df.columns.tolist()
            model_variables = self.variables
            
            # Create a mapping from available data to model variables
            # For simplicity, we'll take the first N columns that match the model size
            if len(available_columns) >= len(model_variables):
                # Use first N columns
                selected_columns = available_columns[:len(model_variables)]
                df_selected = df[selected_columns].copy()
                df_selected.columns = model_variables
            else:
                # Pad with zeros if we don't have enough columns
                df_selected = pd.DataFrame(index=df.index, columns=model_variables)
                for i, col in enumerate(available_columns):
                    if i < len(model_variables):
                        df_selected[model_variables[i]] = df[col]
                df_selected = df_selected.fillna(0)
            
            # Apply differencing (like in training)
            df_differenced = df_selected.diff().dropna()
            
            # Get the most recent data for prediction
            df_recent = df_differenced.tail(self.context_length * 2)  # Extra data for stability
            
            return df_recent
            
        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {e}")
            return None
    
    def save_temporary_data(self, df: pd.DataFrame, cycle: int):
        """Save data temporarily for each cycle"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"temp_data/cycle_{cycle:04d}_{timestamp}.csv"
            df.to_csv(filename)
            self.logger.debug(f"💾 Temporary data saved: {filename}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save temporary data: {e}")
    
    def initialize_dashboard(self):
        """Initialize the Dash dashboard"""
        try:
            dashboard_port = self.config['monitoring']['dashboard_port']
            self.logger.info(f"🔧 Initializing dashboard on port {dashboard_port}...")
            
            # Import here to catch import errors specifically
            from standalone_dashboard import StandaloneDashboard
            
            self.dash_plotter = StandaloneDashboard(port=dashboard_port, debug=False)
            self.logger.info("✅ Dashboard instance created successfully")
            
            self.dash_plotter.start_server()
            self.logger.info(f"🌐 Dashboard started at http://localhost:{dashboard_port}")
            
            # Set connection status
            self.dash_plotter.set_connection_status('Connected')
            
            time.sleep(2)  # Give server time to start
            
        except ImportError as e:
            self.logger.error(f"❌ Dashboard import failed: {e}")
            self.logger.error("   Please install: pip install dash plotly")
            self.dash_plotter = None
        except Exception as e:
            self.logger.error(f"❌ Dashboard initialization failed: {e}")
            import traceback
            self.logger.error(f"   Full error: {traceback.format_exc()}")
            self.dash_plotter = None
    
    def run_prediction_cycle(self, df: pd.DataFrame, cycle: int) -> bool:
        """Run a single prediction cycle"""
        try:
            # Save temporary data for this cycle
            self.save_temporary_data(df, cycle)
            
            # Run the forecasting using your existing function
            predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df = multistep_rolling_buffer_learning_prediction_with_dash(
                initial_model=self.model,
                df_online=df,
                scalers=self.scalers,
                context_length=self.context_length,
                df_removed_nans_forecasting=self.df_removed_nans_forecasting,
                df_removed_nans_classification=self.df_removed_nans_classification,
                dash_plotter=None,  # Use our standalone dashboard instead
                variables=self.variables,
                prediction_horizon=self.prediction_horizon
            )
            
            # Update standalone dashboard with results
            if self.dash_plotter is not None and not predictions_df.empty:
                try:
                    timestamp = datetime.now()
                    
                    # Convert predictions to dictionary format
                    predictions_dict = {}
                    actuals_dict = {}
                    
                    if len(predictions_df.columns) > 0:
                        # Use last row of predictions
                        last_pred = predictions_df.iloc[-1]
                        for col in predictions_df.columns:
                            predictions_dict[col] = float(last_pred[col])
                    
                    if not actuals_df.empty and len(actuals_df.columns) > 0:
                        # Use last row of actuals
                        last_actual = actuals_df.iloc[-1]
                        for col in actuals_df.columns:
                            if col in last_actual:
                                actuals_dict[col] = float(last_actual[col])
                    
                    # Update dashboard
                    self.dash_plotter.update_data(
                        timestamp=timestamp,
                        predictions_dict=predictions_dict,
                        actuals_dict=actuals_dict,
                        is_anomaly=False  # TODO: Add anomaly detection logic
                    )
                    
                    self.logger.debug(f"📊 Dashboard updated with {len(predictions_dict)} predictions")
                    
                except Exception as e:
                    self.logger.error(f"❌ Dashboard update failed: {e}")
            elif self.dash_plotter is None:
                self.logger.debug("DEBUG: dash_plotter is None, dashboard not available")
            else:
                self.logger.debug("DEBUG: predictions_df is empty, skipping dashboard update")
            
            # Log prediction results
            if not predictions_df.empty:
                self.logger.info(f"✅ Cycle {cycle}: Generated {len(predictions_df)} predictions")
                
                # Calculate basic metrics for monitoring
                if not actuals_df.empty and len(predictions_df) == len(actuals_df):
                    mae = np.mean(np.abs(predictions_df.values - actuals_df.values))
                    self.logger.info(f"   MAE: {mae:.6f}")
            else:
                self.logger.warning(f"⚠️ Cycle {cycle}: No predictions generated")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Prediction cycle {cycle} failed: {e}")
            return False
    
    def run_monitoring_loop(self):
        """Main monitoring loop with 1-minute cycles"""
        self.logger.info("🏭 Starting real-time forecasting loop...")
        
        # Initialize connections and models
        if not self.connect_zabbix():
            return False
        
        if not self.load_trained_model():
            return False
        
        # Discover monitoring items
        items = self.discover_monitoring_items()
        if not items:
            return False
        
        # Initialize dashboard
        self.initialize_dashboard()
        
        # Main monitoring loop
        cycle = 0
        self.logger.info(f"⏰ Starting monitoring cycles (every {self.update_interval} seconds)")
        self.logger.info(f"🌐 Dashboard: http://localhost:{self.config['monitoring']['dashboard_port']}")
        
        while True:
            try:
                cycle += 1
                cycle_start = time.time()
                
                self.logger.info(f"🔄 Cycle #{cycle} started")
                
                # Collect current data from Zabbix
                raw_data = self.collect_current_data(items)
                if raw_data is None:
                    self.logger.warning(f"⚠️ Cycle #{cycle}: No data collected, skipping")
                    time.sleep(self.update_interval)
                    continue
                
                # Prepare data for prediction
                prepared_data = self.prepare_data_for_prediction(raw_data)
                if prepared_data is None:
                    self.logger.warning(f"⚠️ Cycle #{cycle}: Data preparation failed, skipping")
                    time.sleep(self.update_interval)
                    continue
                
                # Run prediction cycle
                success = self.run_prediction_cycle(prepared_data, cycle)
                
                cycle_time = time.time() - cycle_start
                
                if success:
                    self.logger.info(f"✅ Cycle #{cycle} completed in {cycle_time:.2f}s")
                else:
                    self.logger.warning(f"⚠️ Cycle #{cycle} had issues ({cycle_time:.2f}s)")
                
                # Wait for next cycle (accounting for processing time)
                sleep_time = max(0, self.update_interval - cycle_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Unexpected error in cycle #{cycle}: {e}")
                time.sleep(self.update_interval)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Zabbix Real-time Forecasting Loop')
    parser.add_argument('--config', '-c', default='config.json', help='Configuration file')
    parser.add_argument('--test', '-t', action='store_true', help='Test connections only')
    
    args = parser.parse_args()
    
    forecaster = ZabbixForecastingLoop(args.config)
    
    if args.test:
        print("🧪 Testing system components...")
        print("=" * 50)
        
        # Test Zabbix connection
        print("1️⃣ Testing Zabbix connection...")
        zabbix_ok = forecaster.connect_zabbix()
        
        # Test model loading
        print("2️⃣ Testing model loading...")
        model_ok = forecaster.load_trained_model()
        
        if zabbix_ok and model_ok:
            # Test item discovery
            print("3️⃣ Testing item discovery...")
            items = forecaster.discover_monitoring_items()
            
            if items:
                print("4️⃣ Testing data collection...")
                data = forecaster.collect_current_data(items)
                if data is not None:
                    print(f"📊 Sample data: {data.shape}")
                    print("✅ All systems ready for monitoring!")
                else:
                    print("⚠️ Data collection test failed")
            else:
                print("⚠️ No monitoring items found")
        else:
            print("❌ System not ready:")
            if not zabbix_ok:
                print("   - Zabbix connection failed")
            if not model_ok:
                print("   - Model loading failed")
        
        print("=" * 50)
        return
    
    # Run monitoring loop
    print("🏭 Starting Zabbix Real-time Forecasting")
    print("   - Press Ctrl+C to stop")
    print(f"   - Dashboard: http://localhost:{forecaster.config['monitoring']['dashboard_port']}")
    print()
    
    forecaster.run_monitoring_loop()


if __name__ == "__main__":
    main()