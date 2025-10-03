#!/usr/bin/env python3
"""
🏭 Optimized Zabbix Data Connector for Industrial Anomaly Detection

Simplified, clean implementation that fetches data from Zabbix and runs ML models.
READ-ONLY integration - no data is sent back to Zabbix.
"""

import sys
import os
import json
import time
import logging
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pyzabbix import ZabbixAPI
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from initial_model import get_initial_model
    from online_forecasting_multi_step import multistep_rolling_buffer_learning_prediction_with_dash
    from dash_plotter import DashRealTimePlotter
    from data_utils import remove_outliers
    from data_preprocessing import Dataset
except ImportError as e:
    print(f"⚠️  Warning: Could not import project modules: {e}")
    print("Ensure you're running from the correct directory")


class OptimizedZabbixConnector:
    """Zabbix connector with same preprocessing as initial model training"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the connector with minimal configuration"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.zabbix_api = None
        
        # Core settings from config
        self.context_length = self.config.get('monitoring', {}).get('context_length', 60)
        self.update_interval = self.config.get('monitoring', {}).get('update_interval', 60)  # 1 minute cycle
        self.prediction_horizon = self.config.get('monitoring', {}).get('prediction_horizon', 6)
        
        # Data storage paths - same structure as training
        self.temp_data_dir = os.path.join(os.path.dirname(__file__), "temp_data")
        os.makedirs(self.temp_data_dir, exist_ok=True)
        
        # Filenames for temporary storage (CSV format like training data)
        self.raw_data_file = os.path.join(self.temp_data_dir, "zabbix_raw_data.csv")
        self.processed_forecasting_file = os.path.join(self.temp_data_dir, "zabbix_forecasting.csv")
        self.processed_classification_file = os.path.join(self.temp_data_dir, "zabbix_classification.csv")
        
        # Variable mappings based on your training data patterns
        # These should match the patterns from your actual training data
        self.variable_mappings = {
            'ICMP response time': ['icmpping', 'icmppingsec', 'ping'],
            'temperature': ['sensor.temp', 'temp', 'temperature'],
            'cpu': ['system.cpu.util', 'cpu.util', 'cpu'],
            'used memory': ['vm.memory.util', 'memory.util', 'memory'],
            'bits': ['net.if.in', 'net.if.out', 'ifInOctets', 'ifOutOctets', 'bits']
        }
        
        # Initialize connection
        self._connect_to_zabbix()
        self.logger.info("🏭 Zabbix Connector initialized with training-compatible preprocessing")

    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required sections
            required_sections = ['zabbix', 'models']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required config section: {section}")
            
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration: {e}")

    def _setup_logging(self) -> logging.Logger:
        """Setup simple logging"""
        logger = logging.getLogger('ZabbixConnector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:  # Avoid duplicate handlers
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _connect_to_zabbix(self):
        """Establish connection to Zabbix API"""
        try:
            zabbix_config = self.config['zabbix']            
            self.zabbix_api = ZabbixAPI(zabbix_config['url'])
            self.zabbix_api.session.verify = False
            
            # Suppress SSL warnings
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            self.zabbix_api.login(zabbix_config['user'], zabbix_config['password'])
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Zabbix: {e}")
            raise ConnectionError(f"Zabbix connection failed: {e}")

    def collect_and_save_raw_data(self, hours_back: int = 2):
        """Collect raw data from Zabbix and save in training-compatible format"""
        try:
            # Get industrial hosts
            hosts = self.get_industrial_hosts()
            if not hosts:
                raise ValueError("No industrial hosts available")
            
            self.logger.info(f"🔄 Collecting raw data from {len(hosts)} hosts...")
            
            # Collect all raw data in training format (name, timestamp, value)
            all_raw_records = []
            
            # Calculate time range
            time_till = int(time.time())
            time_from = time_till - (hours_back * 3600)
            
            for host in hosts:
                # Get all active items for this host
                items = self.zabbix_api.item.get(
                    hostids=[host['hostid']],
                    output=['itemid', 'key_', 'name'],
                    filter={'status': 0}
                )
                
                if not items:
                    continue
                
                # Filter items based on our variable mappings
                relevant_items = {}
                for item in items:
                    for var_name, possible_keys in self.variable_mappings.items():
                        if any(key in item['key_'] for key in possible_keys):
                            # Create unique name combining host and variable type
                            item_name = f"{host['host']} - {var_name} - {item['name']}"
                            relevant_items[item['itemid']] = item_name
                            break
                
                if not relevant_items:
                    continue
                
                # Fetch historical data
                history = self.zabbix_api.history.get(
                    itemids=list(relevant_items.keys()),
                    time_from=time_from,
                    time_till=time_till,
                    output=['itemid', 'clock', 'value'],
                    sortfield='clock'
                )
                
                # Convert to training format
                for record in history:
                    if record['itemid'] in relevant_items:
                        all_raw_records.append({
                            'name': relevant_items[record['itemid']],
                            'timestamp': pd.to_datetime(int(record['clock']), unit='s'),
                            'value': float(record['value'])
                        })
            
            if not all_raw_records:
                raise ValueError("No data retrieved from any host")
            
            # Create DataFrame and save in training format
            raw_df = pd.DataFrame(all_raw_records)
            raw_df.to_csv(self.raw_data_file, index=False)
            
            self.logger.info(f"✅ Raw data saved: {len(all_raw_records)} records to {self.raw_data_file}")
            return raw_df
            
        except Exception as e:
            self.logger.error(f"❌ Raw data collection failed: {e}")
            raise

    def apply_training_preprocessing(self, raw_df: pd.DataFrame = None):
        """Apply the exact same preprocessing as in training pipeline"""
        try:
            # If no raw_df provided, try to load from file
            if raw_df is None:
                if not os.path.exists(self.raw_data_file):
                    raise FileNotFoundError("No raw data file found")
                raw_df = pd.read_csv(self.raw_data_file, parse_dates=['timestamp'])
            
            self.logger.info("🔧 Applying training-compatible preprocessing...")
            
            # Create a temporary Dataset-like structure
            # Save raw data in training format for Dataset class
            temp_file = os.path.join(self.temp_data_dir, "temp_dataset.csv")
            raw_df[['name', 'timestamp', 'value']].to_csv(temp_file, index=False)
            
            # Use the same Dataset preprocessing as in training
            dataset = Dataset(temp_file)
            df_preprocessed = dataset.preprocessing()
            
            if df_preprocessed is None or df_preprocessed.empty:
                raise ValueError("Preprocessing failed - no data returned")
            
            # Apply same filtering as in get_data.py
            # Select only numeric columns (same as training)
            df_numeric = df_preprocessed.select_dtypes(include=[np.number])
            
            # Remove constant columns (same logic as training)
            cols_to_remove = []
            for col in df_numeric.columns:
                if df_numeric[col].nunique() <= 1:
                    cols_to_remove.append(col)
            
            if cols_to_remove:
                self.logger.info(f"Removing constant columns: {cols_to_remove}")
                df_numeric = df_numeric.drop(columns=cols_to_remove)
            
            # Apply outlier removal (same as training)
            df_removed_outliers_forecasting = remove_outliers(df_numeric, threshold=1000)
            df_removed_nans_forecasting = df_removed_outliers_forecasting.dropna(axis=1, how="all")
            
            # For classification (same data in this case)
            df_removed_nans_classification = df_removed_nans_forecasting.copy()
            
            # Save processed data (same format as training)
            df_removed_nans_forecasting.to_csv(self.processed_forecasting_file)
            df_removed_nans_classification.to_csv(self.processed_classification_file)
            
            self.logger.info(f"✅ Preprocessing completed:")
            self.logger.info(f"   📊 Forecasting data shape: {df_removed_nans_forecasting.shape}")
            self.logger.info(f"   💾 Saved to: {self.processed_forecasting_file}")
            
            # Clean up temp file
            os.remove(temp_file)
            
            return df_removed_nans_forecasting, df_removed_nans_classification
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing failed: {e}")
            raise

    def get_industrial_hosts(self) -> List[Dict]:
        """Get industrial devices from configured host groups"""
        try:
            device_groups = self.config.get('industrial_filters', {}).get(
                'device_groups', ["Virtual machines", "Zabbix servers"]
            )
            
            # Get host groups
            groups = self.zabbix_api.hostgroup.get(
                filter={'name': device_groups},
                output=['groupid', 'name']
            )
            
            if not groups:
                self.logger.warning(f"⚠️  No groups found matching: {device_groups}")
                return []
            
            group_ids = [group['groupid'] for group in groups]
            
            # Get enabled hosts in these groups
            hosts = self.zabbix_api.host.get(
                groupids=group_ids,
                output=['hostid', 'host', 'name'],
                filter={'status': 0}  # Only enabled hosts
            )
            
            self.logger.info(f"🏭 Found {len(hosts)} industrial hosts")
            return hosts
            
        except Exception as e:
            self.logger.error(f"❌ Host discovery failed: {e}")
            return []

    def get_training_data_format(self, hours_back: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, int, List[str]]:
        """
        Get data in the exact format expected by your training pipeline using same preprocessing
        Returns: (df_online, df_removed_nans_forecasting, scalers, context_length, variables)
        """
        try:
            self.logger.info(f"🔄 Collecting and preprocessing {hours_back} hours of data...")
            
            # Step 1: Collect raw data from Zabbix (same format as training CSV)
            raw_df = self.collect_and_save_raw_data(hours_back)
            
            # Step 2: Apply the exact same preprocessing as in training
            df_removed_nans_forecasting, df_removed_nans_classification = self.apply_training_preprocessing(raw_df)
            
            # Step 3: Apply differencing (same as initial_model.py)
            df_differenced = df_removed_nans_forecasting.diff().dropna()
            
            # Step 4: Create df_online from the processed data (same as training)
            df_online = df_differenced.copy()
            variables = df_online.columns.tolist()
            
            # Step 5: Load or create scalers (same approach as training)
            scalers = self._load_or_create_scalers(df_online, variables)
            
            self.logger.info(f"✅ Data processed with training pipeline:")
            self.logger.info(f"   📊 df_online shape: {df_online.shape}")
            self.logger.info(f"   🏷️  Variables: {len(variables)}")
            self.logger.info(f"   📅 Time range: {df_online.index.min()} to {df_online.index.max()}")
            
            return df_online, df_removed_nans_forecasting, scalers, self.context_length, variables
            
        except Exception as e:
            self.logger.error(f"❌ Data processing failed: {e}")
            raise

    def _load_or_create_scalers(self, df: pd.DataFrame, variables: List[str]) -> Dict:
        """Load scalers from trained model or create new ones"""
        model_path = self.config.get('models', {}).get('forecasting_model_path')
        scalers_file = os.path.join(model_path, 'scalers_train.pkl') if model_path else None
        
        if scalers_file and os.path.exists(scalers_file):
            with open(scalers_file, 'rb') as f:
                scalers = pickle.load(f)
            self.logger.info("✅ Loaded scalers from trained model")
            return scalers
        else:
            # Create new scalers for current data
            scalers = {}
            for var in variables:
                scaler = StandardScaler()
                data = df[var].dropna().values.reshape(-1, 1)
                if len(data) > 0:
                    scaler.fit(data)
                    scalers[var] = scaler
            
            self.logger.warning("⚠️  Created new scalers (trained scalers not found)")
            return scalers

    def run_monitoring_loop(self):
        """Main monitoring loop - simplified and clean"""
        try:
            self.logger.info("🚀 Starting optimized monitoring loop...")
            
            # Load trained model
            model_path = self.config.get('models', {}).get('forecasting_model_path')
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            initial_model = get_initial_model(model_path)
            self.logger.info("📦 Loaded forecasting model")
            
            # Initialize dashboard
            plotter = DashRealTimePlotter()
            plotter.start_server()
            self.logger.info("🎯 Dashboard started at http://localhost:8050")
            time.sleep(3)  # Allow initialization
            
            # Main loop
            cycle_count = 0
            while True:
                try:
                    cycle_count += 1
                    self.logger.info(f"🔄 Starting monitoring cycle #{cycle_count}")
                    
                    # Get fresh data from Zabbix
                    df_online, df_removed_nans_forecasting, scalers, context_length, variables = self.get_training_data_format()
                    
                    if df_online.empty:
                        self.logger.warning("⚠️  No data received, skipping cycle...")
                        time.sleep(self.update_interval)
                        continue
                    
                    # Prepare data structures for your existing forecasting function
                    df_removed_nans_classification = df_removed_nans_forecasting.copy()
                    
                    # Run forecasting using your existing pipeline
                    self.logger.info(f"🎯 Running forecasting on {len(df_online)} data points...")
                    
                    predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df = \
                        multistep_rolling_buffer_learning_prediction_with_dash(
                            initial_model=initial_model,
                            df_online=df_online,
                            scalers=scalers,
                            context_length=context_length,
                            df_removed_nans_forecasting=df_removed_nans_forecasting,
                            df_removed_nans_classification=df_removed_nans_classification,
                            dash_plotter=plotter,
                            variables=variables,
                            prediction_horizon=self.prediction_horizon
                        )
                    
                    self.logger.info(f"✅ Cycle #{cycle_count} completed successfully")
                    
                    # Wait for next cycle
                    self.logger.info(f"⏱️  Waiting {self.update_interval}s for next cycle...")
                    time.sleep(self.update_interval)
                    
                except KeyboardInterrupt:
                    self.logger.info("🛑 Shutdown requested")
                    break
                except Exception as e:
                    self.logger.error(f"❌ Error in cycle #{cycle_count}: {e}")
                    self.logger.info(f"⏱️  Retrying in {self.update_interval}s...")
                    time.sleep(self.update_interval)
            
        except Exception as e:
            self.logger.error(f"❌ Fatal error in monitoring loop: {e}")
            raise

    def test_connection(self):
        """Test Zabbix connection and data retrieval"""
        try:
            self.logger.info("🧪 Testing Zabbix connection...")
            
            # Test data retrieval
            df_online, df_removed_nans_forecasting, scalers, context_length, variables = self.get_training_data_format(hours_back=1)
            
            print(f"✅ Connection test successful!")
            print(f"   📊 Data shape: {df_online.shape}")
            print(f"   🏷️  Variables: {len(variables)}")
            print(f"   📅 Time range: {df_online.index.min()} to {df_online.index.max()}")
            print(f"   ⚙️  Context length: {context_length}")
            print(f"   🔧 Scalers loaded: {len(scalers)}")
            
            if len(variables) > 0:
                print(f"   📋 Sample variables: {variables[:3]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False


def main():
    """Main entry point with simplified argument handling"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimized Zabbix Connector for Industrial Anomaly Detection'
    )
    parser.add_argument(
        '--config', '-c', 
        default='config.json',
        help='Configuration file path (default: config.json)'
    )
    parser.add_argument(
        '--test', '-t', 
        action='store_true',
        help='Run connection test only'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize connector
        connector = OptimizedZabbixConnector(args.config)
        
        if args.test:
            # Test mode
            success = connector.test_connection()
            sys.exit(0 if success else 1)
        else:
            # Production monitoring
            print("🏭 Starting production monitoring...")
            print("Press Ctrl+C to stop")
            connector.run_monitoring_loop()
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown completed")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()