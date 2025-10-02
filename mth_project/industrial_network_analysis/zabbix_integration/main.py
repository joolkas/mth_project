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
except ImportError as e:
    print(f"⚠️  Warning: Could not import project modules: {e}")
    print("Ensure you're running from the correct directory")


class OptimizedZabbixConnector:
    """Simplified Zabbix connector for industrial anomaly detection"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the connector with minimal configuration"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.zabbix_api = None
        
        # Core settings from config
        self.context_length = self.config.get('monitoring', {}).get('context_length', 60)
        self.update_interval = self.config.get('monitoring', {}).get('update_interval', 60)
        self.prediction_horizon = self.config.get('monitoring', {}).get('prediction_horizon', 6)
        
        # Variable mappings (simplified from your training data)
        self.variable_mappings = {
            'ICMP response time': ['icmpping', 'icmppingsec'],
            'Switch 1 - Temperature': ['sensor.temp.1', 'temp.1', 'temperature.1'],
            'Switch 2 - Temperature': ['sensor.temp.2', 'temp.2', 'temperature.2'],
            'CPU utilization': ['system.cpu.util', 'cpu.util'],
            'Memory utilization': ['vm.memory.util', 'memory.util'],
            'Interface Bits received': ['net.if.in'],
            'Interface Bits sent': ['net.if.out']
        }
        
        # Initialize connection
        self._connect_to_zabbix()
        self.logger.info("🏭 Optimized Zabbix Connector initialized")

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
            self.zabbix_api.login(zabbix_config['user'], zabbix_config['password'])
            
            # Verify connection
            version = self.zabbix_api.apiinfo.version()
            self.logger.info(f"✅ Connected to Zabbix API v{version}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Zabbix: {e}")
            raise ConnectionError(f"Zabbix connection failed: {e}")

    def get_industrial_hosts(self) -> List[Dict]:
        """Get industrial devices from configured host groups"""
        try:
            device_groups = self.config.get('industrial_filters', {}).get(
                'device_groups', ['Industrial', 'SCADA', 'PLC']
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

    def fetch_host_metrics(self, host_id: str, hours_back: int = 2) -> pd.DataFrame:
        """Fetch metrics for a single host and map to training variables"""
        try:
            # Calculate time range
            time_till = int(time.time())
            time_from = time_till - (hours_back * 3600)
            
            # Get all active items for this host
            items = self.zabbix_api.item.get(
                hostids=[host_id],
                output=['itemid', 'key_', 'name'],
                filter={'status': 0}
            )
            
            if not items:
                return pd.DataFrame()
            
            # Map items to training variables
            item_mapping = {}
            for item in items:
                for var_name, possible_keys in self.variable_mappings.items():
                    if any(key in item['key_'] for key in possible_keys):
                        item_mapping[item['itemid']] = var_name
                        break
            
            if not item_mapping:
                return pd.DataFrame()
            
            # Fetch historical data
            history = self.zabbix_api.history.get(
                itemids=list(item_mapping.keys()),
                time_from=time_from,
                time_till=time_till,
                output=['itemid', 'clock', 'value'],
                sortfield='clock'
            )
            
            if not history:
                return pd.DataFrame()
            
            # Convert to DataFrame
            records = []
            for record in history:
                if record['itemid'] in item_mapping:
                    records.append({
                        'timestamp': pd.to_datetime(int(record['clock']), unit='s'),
                        'variable': item_mapping[record['itemid']],
                        'value': float(record['value'])
                    })
            
            if not records:
                return pd.DataFrame()
            
            # Pivot and resample
            df = pd.DataFrame(records)
            df_pivot = df.pivot_table(
                index='timestamp',
                columns='variable',
                values='value',
                aggfunc='first'
            )
            
            # Resample to 1-minute intervals and fill missing values
            df_resampled = df_pivot.resample('1T').mean()
            df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')
            
            return df_resampled
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch metrics for host {host_id}: {e}")
            return pd.DataFrame()

    def get_training_data_format(self, hours_back: int = 2) -> Tuple[pd.DataFrame, Dict, int, List[str]]:
        """
        Get data in the exact format expected by your training pipeline
        Returns: (df_online, scalers, context_length, variables)
        """
        try:
            # Get industrial hosts
            hosts = self.get_industrial_hosts()
            if not hosts:
                raise ValueError("No industrial hosts available")
            
            self.logger.info(f"🔄 Fetching {hours_back} hours of data from {len(hosts)} hosts...")
            
            # Collect data from all hosts
            host_dataframes = []
            for host in hosts:
                host_data = self.fetch_host_metrics(host['hostid'], hours_back)
                if not host_data.empty:
                    # Add host prefix to distinguish variables from different hosts
                    host_data.columns = [f"{host['host']}_{col}" for col in host_data.columns]
                    host_dataframes.append(host_data)
            
            if not host_dataframes:
                raise ValueError("No data retrieved from any host")
            
            # Combine all host data
            df_combined = pd.concat(host_dataframes, axis=1, sort=True)
            
            # Ensure 1-minute intervals and handle missing data
            df_final = df_combined.resample('1T').mean()
            df_final = df_final.fillna(method='ffill').fillna(method='bfill')
            
            # Get variable list
            variables = list(df_final.columns)
            
            # Load or create scalers
            scalers = self._load_or_create_scalers(df_final, variables)
            
            self.logger.info(f"✅ Retrieved {len(df_final)} data points, {len(variables)} variables")
            return df_final, scalers, self.context_length, variables
            
        except Exception as e:
            self.logger.error(f"❌ Data retrieval failed: {e}")
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
                    df_online, scalers, context_length, variables = self.get_training_data_format()
                    
                    if df_online.empty:
                        self.logger.warning("⚠️  No data received, skipping cycle...")
                        time.sleep(self.update_interval)
                        continue
                    
                    # Prepare data structures for your existing forecasting function
                    df_removed_nans_forecasting = df_online.copy()
                    df_removed_nans_classification = df_online.copy()
                    
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
            df_online, scalers, context_length, variables = self.get_training_data_format(hours_back=1)
            
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