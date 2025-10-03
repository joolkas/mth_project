#!/usr/bin/env python3
"""
🏭 Simple Zabbix Industrial Monitoring

Ultra-simplified version that automatically adapts to any Zabbix environment.
Fetches data, adapts features, runs ML models - all automatically.
"""

import sys
import os
import json
import time
import logging
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pyzabbix import ZabbixAPI
from sklearn.preprocessing import StandardScaler
import urllib3
urllib3.disable_warnings()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from initial_model import get_initial_model
    from online_forecasting_multi_step import multistep_rolling_buffer_learning_prediction_with_dash
    from data_utils import remove_outliers
    from data_preprocessing import Dataset
except ImportError as e:
    print(f"⚠️ Import error: {e}")


class SimpleZabbixMonitor:
    """Ultra-simple Zabbix monitoring with auto-adaptation"""
    
    def __init__(self, config_path="config.json"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.zabbix_api = None
        self.model = None
        self.scalers = {}
        self.variables = []
        
        # Simple settings
        self.update_interval = 60  # 1 minute
        self.context_length = 60
        
    def _load_config(self, config_path):
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Config error: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """Simple logging setup"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def connect_zabbix(self):
        """Connect to Zabbix API"""
        try:
            zabbix_config = self.config['zabbix']
            self.zabbix_api = ZabbixAPI(zabbix_config['url'])
            self.zabbix_api.session.verify = False
            self.zabbix_api.login(zabbix_config['user'], zabbix_config['password'])
            self.logger.info("✅ Connected to Zabbix")
            return True
        except Exception as e:
            self.logger.error(f"❌ Zabbix connection failed: {e}")
            return False
    
    def load_model(self):
        """Load the ML model"""
        try:
            model_path = self.config['models']['forecasting_model_path']
            self.model = get_initial_model(model_path)
            
            # Load scalers if available
            scalers_path = os.path.join(model_path, 'scalers.pkl')
            if os.path.exists(scalers_path):
                with open(scalers_path, 'rb') as f:
                    self.scalers = pickle.load(f)
            
            self.logger.info("✅ Model loaded")
            return True
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def get_zabbix_data(self):
        """Get current data from Zabbix - automatically discovers everything"""
        try:
            # Get all hosts from configured groups
            host_groups = self.config['industrial_filters'].get('device_groups', [])
            all_hosts = []
            
            for group_name in host_groups:
                groups = self.zabbix_api.hostgroup.get(filter={"name": group_name})
                if groups:
                    hosts = self.zabbix_api.host.get(groupids=[groups[0]['groupid']])
                    all_hosts.extend(hosts)
            
            if not all_hosts:
                self.logger.warning("No hosts found")
                return None
            
            # Get recent data (last hour)
            time_to = int(time.time())
            time_from = time_to - 3600  # 1 hour
            
            all_data = []
            
            for host in all_hosts:
                host_name = host['host']
                
                # Get all monitored items
                items = self.zabbix_api.item.get(
                    hostids=[host['hostid']],
                    output=['itemid', 'name'],
                    monitored=True
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
                
                # Convert to DataFrame format
                item_names = {item['itemid']: f"{host_name} - {item['name']}" for item in items}
                
                for record in history:
                    if record['itemid'] in item_names:
                        all_data.append({
                            'name': item_names[record['itemid']],
                            'timestamp': pd.to_datetime(int(record['clock']), unit='s'),
                            'value': float(record['value'])
                        })
            
            if not all_data:
                return None
            
            # Create time series DataFrame
            df = pd.DataFrame(all_data)
            df_pivot = df.pivot_table(
                index='timestamp',
                columns='name', 
                values='value',
                aggfunc='mean'
            ).fillna(method='ffill').fillna(0)
            
            self.logger.info(f"📊 Collected data: {df_pivot.shape}")
            return df_pivot
            
        except Exception as e:
            self.logger.error(f"❌ Data collection error: {e}")
            return None
    
    def auto_adapt_features(self, df, expected_features):
        """Automatically adapt features to match model"""
        current_features = len(df.columns)
        
        if current_features == expected_features:
            return df  # Perfect match
        
        if current_features > expected_features:
            # Select most important features
            self.logger.info(f"🔧 Auto-selecting {expected_features} from {current_features} features")
            
            # Score columns by importance keywords
            importance_keywords = [
                'memory', 'Memory', 'cpu', 'CPU', 'utilization',
                'Interface', 'Bits', 'network', 'traffic',
                'Space', 'Available', 'Used', 'Total',
                'temperature', 'Temperature'
            ]
            
            scores = []
            for col in df.columns:
                score = sum(1 for keyword in importance_keywords if keyword in col)
                scores.append((score, col))
            
            # Select top features
            scores.sort(reverse=True)
            selected_cols = [col for _, col in scores[:expected_features]]
            
            self.variables = selected_cols
            return df[selected_cols]
        
        else:
            # Too few features - can't proceed
            self.logger.error(f"❌ Too few features: need {expected_features}, have {current_features}")
            return None
    
    def save_current_variables(self, variables):
        """Save current variables to file for future reference"""
        try:
            variables_file = os.path.join(self.config['models']['forecasting_model_path'], 'variables.txt')
            with open(variables_file, 'w') as f:
                for var in variables:
                    f.write(f"{var}\n")
            self.logger.info(f"💾 Saved {len(variables)} variables to {variables_file}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not save variables: {e}")
    
    def run_prediction(self, df):
        """Run ML prediction on the data"""
        try:
            # Ensure we have enough data
            if len(df) < self.context_length:
                self.logger.warning(f"⚠️ Need {self.context_length} records, have {len(df)}")
                return None
            
            # Simple scaling if no scalers available
            if not self.scalers:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                df_scaled = pd.DataFrame(
                    scaler.fit_transform(df),
                    columns=df.columns,
                    index=df.index
                )
                self.scalers = {i: scaler for i in range(len(df.columns))}
            else:
                df_scaled = df.copy()
                for i, col in enumerate(df.columns):
                    if i in self.scalers:
                        df_scaled[col] = self.scalers[i].transform(df[[col]])
            
            # Run the existing forecasting function
            predictions_df, actuals_df, _, _ = multistep_rolling_buffer_learning_prediction_with_dash(
                initial_model=self.model,
                df_online=df_scaled,
                scalers=self.scalers,
                context_length=self.context_length,
                df_removed_nans_forecasting=df_scaled,
                df_removed_nans_classification=df_scaled.copy()
            )
            
            self.logger.info("✅ Predictions completed")
            return predictions_df, actuals_df
            
        except Exception as e:
            self.logger.error(f"❌ Prediction error: {e}")
            return None
    
    def run_monitoring(self):
        """Main monitoring loop - ultra simple"""
        
        # Initialize
        if not self.connect_zabbix():
            return
        
        if not self.load_model():
            return
        
        # Get expected features from model
        try:
            expected_features = self.model.input_shape[2] if len(self.model.input_shape) == 3 else 23
        except:
            expected_features = 23  # Default fallback
        
        self.logger.info(f"🎯 Starting monitoring (expecting {expected_features} features)")
        
        cycle = 0
        while True:
            try:
                cycle += 1
                self.logger.info(f"🔄 Cycle #{cycle}")
                
                # Get fresh data
                df = self.get_zabbix_data()
                if df is None or df.empty:
                    self.logger.warning("⚠️ No data, skipping cycle")
                    time.sleep(self.update_interval)
                    continue
                
                # Auto-adapt features
                df_adapted = self.auto_adapt_features(df, expected_features)
                if df_adapted is None:
                    time.sleep(self.update_interval)
                    continue
                
                # Save current variables for reference
                if hasattr(self, 'variables') and self.variables:
                    self.save_current_variables(self.variables)
                
                # Run prediction
                result = self.run_prediction(df_adapted)
                if result:
                    self.logger.info(f"✅ Cycle #{cycle} completed successfully")
                else:
                    self.logger.warning(f"⚠️ Cycle #{cycle} failed")
                
                # Wait for next cycle
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Cycle error: {e}")
                time.sleep(self.update_interval)


def main():
    """Simple main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Zabbix Industrial Monitoring')
    parser.add_argument('--config', '-c', default='config.json', help='Config file')
    parser.add_argument('--test', '-t', action='store_true', help='Test connection only')
    
    args = parser.parse_args()
    
    monitor = SimpleZabbixMonitor(args.config)
    
    if args.test:
        print("🧪 Testing connection...")
        success = monitor.connect_zabbix() and monitor.load_model()
        if success:
            print("✅ Test successful!")
            data = monitor.get_zabbix_data()
            if data is not None:
                print(f"📊 Sample data: {data.shape}")
                print(f"🏷️ Sample columns: {list(data.columns)[:5]}")
        sys.exit(0 if success else 1)
    else:
        print("🏭 Starting Simple Zabbix Monitoring...")
        print("Press Ctrl+C to stop")
        monitor.run_monitoring()


if __name__ == "__main__":
    main()