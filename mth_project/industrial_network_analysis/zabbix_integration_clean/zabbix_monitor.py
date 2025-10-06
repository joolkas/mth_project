#!/usr/bin/env python3
"""
🏭 Minimal Zabbix Integration for Industrial Anomaly Detection

Simplified READ-ONLY integration that fetches data from Zabbix and runs
your existing anomaly detection models without modification.

Key Features:
- Connects to Zabbix API (read-only)
- Fetches industrial device metrics
- Runs existing LSTM forecasting
- Displays results on Dash dashboard
- No data sent back to Zabbix

Author: Industrial Network Analysis System
"""

import sys
import os
import json
import time
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pyzabbix import ZabbixAPI
import urllib3

# Disable SSL warnings for internal networks
urllib3.disable_warnings()

# Add parent directory to import the original modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from initial_model import get_initial_model, get_online_data
    from online_forecasting_multi_step import multistep_rolling_buffer_learning_prediction_with_dash
    from dash_plotter import DashRealTimePlotter
except ImportError as e:
    print(f"❌ Cannot import required modules: {e}")
    print("   Make sure this script is in the zabbix_integration directory")
    sys.exit(1)


class ZabbixAnomalyMonitor:
    """Minimal Zabbix connector for anomaly detection"""
    
    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        self.zabbix_api = None
        self.model = None
        self.dash_plotter = None
        
        # System parameters (matching your original system)
        self.update_interval = 60  # 1 minute monitoring cycle
        self.context_length = 60   # Same as original system
        self.prediction_horizon = 6  # Same as original system
        
    def _load_config(self, config_file: str) -> dict:
        """Load minimal configuration"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                # Create default config if not exists
                default_config = {
                    "zabbix": {
                        "url": "http://localhost/zabbix",
                        "user": "Admin", 
                        "password": "zabbix"
                    },
                    "models": {
                        "forecasting_model_path": "../forecasting_model"
                    },
                    "monitoring": {
                        "context_length": 60,
                        "update_interval": 60,
                        "prediction_horizon": 6
                    },
                    "industrial_filters": {
                        "device_groups": ["Zabbix servers", "Linux servers"]
                    }
                }
                
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                print(f"📝 Created default config: {config_file}")
                print("   Please update with your Zabbix credentials")
                return default_config
                
        except Exception as e:
            print(f"❌ Config error: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """Simple logging setup"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('zabbix_monitor.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def connect_zabbix(self) -> bool:
        """Connect to Zabbix API"""
        try:
            zabbix_config = self.config['zabbix']
            print(f"🔗 Attempting to connect to: {zabbix_config['url']}")
            print(f"👤 Using user: {zabbix_config['user']}")
            
            self.zabbix_api = ZabbixAPI(zabbix_config['url'])
            self.zabbix_api.session.verify = False  # For internal networks
            
            print("🔐 Attempting login...")
            self.zabbix_api.login(zabbix_config['user'], zabbix_config['password'])
            
            # Test connection
            print("📡 Testing API connection...")
            version = self.zabbix_api.apiinfo.version()
            print(f"✅ Connected to Zabbix {version}")
            self.logger.info(f"✅ Connected to Zabbix {version}")
            return True
            
        except Exception as e:
            print(f"❌ Zabbix connection failed: {e}")
            print(f"   URL: {zabbix_config.get('url', 'Not set')}")
            print(f"   User: {zabbix_config.get('user', 'Not set')}")
            print("   Please check:")
            print("   1. Zabbix server is running and accessible")
            print("   2. URL is correct (http://<server>/zabbix)")
            print("   3. Username and password are correct")
            print("   4. User has API access permissions")
            self.logger.error(f"❌ Zabbix connection failed: {e}")
            return False
    
    def load_models(self) -> bool:
        """Load your existing trained models"""
        try:
            model_path = self.config['models']['forecasting_model_path']
            
            # Load the forecasting model using your existing function
            self.model = get_initial_model(model_path)
            
            self.logger.info("✅ Forecasting model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def get_industrial_data(self, hours_back: int = 2) -> Optional[pd.DataFrame]:
        """Fetch recent data from industrial devices"""
        try:
            # Get hosts from configured groups
            host_groups = self.config.get('industrial_filters', {}).get('device_groups', ['Zabbix servers'])
            all_hosts = []
            
            for group_name in host_groups:
                try:
                    groups = self.zabbix_api.hostgroup.get(filter={"name": group_name})
                    if groups:
                        hosts = self.zabbix_api.host.get(
                            groupids=[groups[0]['groupid']],
                            output=['hostid', 'host', 'name']
                        )
                        all_hosts.extend(hosts)
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not get hosts from group {group_name}: {e}")
            
            if not all_hosts:
                self.logger.warning("⚠️ No industrial hosts found")
                return None
            
            # Get recent data 
            time_to = int(time.time())
            time_from = time_to - (hours_back * 3600)
            
            all_data = []
            
            for host in all_hosts:
                try:
                    # Get monitored items from this host
                    items = self.zabbix_api.item.get(
                        hostids=[host['hostid']],
                        output=['itemid', 'name', 'key_'],
                        monitored=True,
                        filter={'value_type': [0, 3]}  # Numeric values only
                    )
                    
                    if not items:
                        continue
                    
                    # Get history for all items
                    item_ids = [item['itemid'] for item in items]
                    history = self.zabbix_api.history.get(
                        itemids=item_ids,
                        time_from=time_from,
                        time_till=time_to,
                        output='extend',
                        sortfield='clock'
                    )
                    
                    # Process history data
                    item_names = {
                        item['itemid']: f"{host['host']}_{item['name']}" 
                        for item in items
                    }
                    
                    for record in history:
                        if record['itemid'] in item_names:
                            try:
                                all_data.append({
                                    'timestamp': pd.to_datetime(int(record['clock']), unit='s'),
                                    'variable': item_names[record['itemid']],
                                    'value': float(record['value'])
                                })
                            except (ValueError, TypeError):
                                continue  # Skip invalid values
                                
                except Exception as e:
                    self.logger.warning(f"⚠️ Error getting data from host {host['host']}: {e}")
                    continue
            
            if not all_data:
                self.logger.warning("⚠️ No numeric data collected")
                return None
            
            # Create time series DataFrame
            df = pd.DataFrame(all_data)
            df_pivot = df.pivot_table(
                index='timestamp',
                columns='variable',
                values='value',
                aggfunc='mean'
            )
            
            # Clean the data
            df_pivot = df_pivot.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            self.logger.info(f"📊 Collected data: {df_pivot.shape} from {len(all_hosts)} hosts")
            return df_pivot
            
        except Exception as e:
            self.logger.error(f"❌ Data collection error: {e}")
            return None
    
    def prepare_data_for_model(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare Zabbix data to match your model's expected format"""
        try:
            if df is None or df.empty:
                return None
            
            # Ensure we have enough historical data
            if len(df) < self.context_length:
                self.logger.warning(f"⚠️ Need at least {self.context_length} records, have {len(df)}")
                return None
            
            # Sort by timestamp and ensure regular intervals
            df = df.sort_index()
            
            # Select most recent data for processing
            df_recent = df.tail(self.context_length * 2)  # Get extra data for stability
            
            # Resample to 1-minute intervals (matching your training data)
            df_resampled = df_recent.resample('1T').mean().fillna(method='ffill')
            
            # Select numeric columns only
            df_numeric = df_resampled.select_dtypes(include=[np.number])
            
            # Drop columns with all NaN or constant values
            df_clean = df_numeric.dropna(axis=1, how='all')
            df_clean = df_clean.loc[:, df_clean.std() > 1e-6]  # Remove constant columns
            
            self.logger.info(f"✅ Prepared data: {df_clean.shape}")
            return df_clean
            
        except Exception as e:
            self.logger.error(f"❌ Data preparation error: {e}")
            return None
    
    def run_anomaly_detection(self, df: pd.DataFrame) -> bool:
        """Run your existing anomaly detection system"""
        try:
            if df is None or df.empty:
                return False
            
            # Load online data format (scalers, etc.) from your saved model
            model_path = self.config['models']['forecasting_model_path']
            try:
                df_online, scalers, context_length, df_removed_nans_forecasting, df_removed_nans_classification, variables, model_mode = get_online_data(model_path)
                self.logger.info(f"📋 Using saved model parameters: {len(variables)} variables")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load saved online data: {e}")
                # Create minimal scalers and parameters
                from sklearn.preprocessing import StandardScaler
                scalers = {i: StandardScaler().fit(df.iloc[:, [i]]) for i in range(min(len(df.columns), 23))}
                variables = df.columns[:23].tolist()  # Limit to reasonable number
                df_removed_nans_forecasting = df.copy()
                df_removed_nans_classification = df.copy()
            
            # Ensure data matches expected dimensions
            if len(df.columns) > len(variables):
                df = df.iloc[:, :len(variables)]
                df.columns = variables[:len(df.columns)]
            elif len(df.columns) < len(variables):
                # Pad with zeros if needed
                for i in range(len(df.columns), len(variables)):
                    df[f'padding_{i}'] = 0
                df.columns = variables[:len(df.columns)]
            
            # Initialize Dash plotter if not already done
            if self.dash_plotter is None:
                self.dash_plotter = DashRealTimePlotter()
                self.dash_plotter.start_server()
                self.logger.info("🌐 Dash server started at http://localhost:8050")
                time.sleep(2)  # Give server time to start
            
            # Run your existing multi-step forecasting function
            predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df = multistep_rolling_buffer_learning_prediction_with_dash(
                initial_model=self.model,
                df_online=df,
                scalers=scalers,
                context_length=self.context_length,
                df_removed_nans_forecasting=df_removed_nans_forecasting,
                df_removed_nans_classification=df_removed_nans_classification,
                dash_plotter=self.dash_plotter,
                variables=variables,
                prediction_horizon=self.prediction_horizon
            )
            
            self.logger.info("✅ Anomaly detection completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Anomaly detection error: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def run_monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info("🏭 Starting industrial anomaly monitoring...")
        
        # Initialize connections and models
        if not self.connect_zabbix():
            return False
        
        if not self.load_models():
            return False
        
        self.logger.info(f"⏰ Monitoring every {self.update_interval} seconds")
        self.logger.info("🌐 Dashboard will be available at http://localhost:8050")
        
        cycle = 0
        while True:
            try:
                cycle += 1
                self.logger.info(f"🔄 Monitoring cycle #{cycle}")
                
                # Get fresh data from Zabbix
                raw_data = self.get_industrial_data(hours_back=2)
                if raw_data is None:
                    self.logger.warning("⚠️ No data collected, skipping cycle")
                    time.sleep(self.update_interval)
                    continue
                
                # Prepare data for your model
                prepared_data = self.prepare_data_for_model(raw_data)
                if prepared_data is None:
                    self.logger.warning("⚠️ Data preparation failed, skipping cycle")
                    time.sleep(self.update_interval)
                    continue
                
                # Run anomaly detection
                success = self.run_anomaly_detection(prepared_data)
                if success:
                    self.logger.info(f"✅ Cycle #{cycle} completed successfully")
                else:
                    self.logger.warning(f"⚠️ Cycle #{cycle} had issues")
                
                # Wait for next cycle
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Unexpected error in cycle #{cycle}: {e}")
                time.sleep(self.update_interval)  # Continue monitoring despite errors


def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Zabbix Industrial Anomaly Monitor')
    parser.add_argument('--config', '-c', default='config.json', help='Configuration file')
    parser.add_argument('--test', '-t', action='store_true', help='Test connections only')
    
    args = parser.parse_args()
    
    monitor = ZabbixAnomalyMonitor(args.config)
    
    if args.test:
        print("🧪 Testing connections...")
        print("=" * 50)
        
        print("\n1️⃣ Testing Zabbix Connection...")
        zabbix_ok = monitor.connect_zabbix()
        
        print("\n2️⃣ Testing Model Loading...")
        models_ok = monitor.load_models()
        
        if zabbix_ok and models_ok:
            print("\n3️⃣ Testing Data Collection...")
            # Test data collection
            data = monitor.get_industrial_data(hours_back=1)
            if data is not None:
                print(f"📊 Sample data collected: {data.shape}")
                print(f"🏷️ Sample variables: {list(data.columns)[:5]}...")
                print("✅ All systems ready!")
            else:
                print("⚠️ No data collected - check host groups and monitored items")
                configured_groups = monitor.config.get('industrial_filters', {}).get('device_groups', ['Zabbix servers'])
                print(f"   Configured host groups: {configured_groups}")
                
        else:
            print("\n❌ System not ready:")
            if not zabbix_ok:
                print("   - Zabbix connection failed")
            if not models_ok:
                print("   - Model loading failed")
        
        print("=" * 50)
        return
    
    # Run monitoring
    print("🏭 Starting Zabbix Industrial Anomaly Detection")
    print("   - Press Ctrl+C to stop")
    print("   - Dashboard: http://localhost:8050")
    print()
    
    monitor.run_monitoring_loop()


if __name__ == "__main__":
    main()