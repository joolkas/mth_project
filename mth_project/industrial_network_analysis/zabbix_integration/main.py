#!/usr/bin/env python3#!/usr/bin/env python3

""""""

🏭 Zabbix Data Connector for Industrial Anomaly Detection🏭 Zabbix Data Connector for Industrial Anomaly Detection

Simple READ-ONLY integration with existing online forecasting loopIntegrates with existing online forecasting loop - READ-ONLY mode



Fetches data FROM Zabbix and runs your trained ML modelsThis connector:

No data sent back to Zabbix - pure monitoring mode1. Fetches data FROM Zabbix (no sending back to Zabbix)

"""2. Replaces df_online with real industrial device data

3. Maintains 1-minute monitoring cycle

import sys4. Preserves compatibility with existing models and preprocessing

import os"""

import json

import timeimport sys

import loggingimport os

import numpy as npimport json

import pandas as pdimport time

from typing import Dict, List, Tupleimport sqlite3

import pickleimport logging

import threading

# Zabbix API clientimport numpy as np

from pyzabbix import ZabbixAPIimport pandas as pd

from datetime import datetime, timedelta

# ML dependenciesfrom typing import Dict, List, Optional, Tuple

import tensorflow as tfimport warnings

from sklearn.preprocessing import StandardScaler

# Zabbix API client

# Import your existing modulestry:

sys.path.append('/opt/anomaly_detection')    from pyzabbix import ZabbixAPI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))except ImportError:

    print("❌ pyzabbix not installed. Install with: pip install pyzabbix")

from data_utils import *    sys.exit(1)

from initial_model import get_initial_model

from online_forecasting_multi_step import multistep_rolling_buffer_learning_prediction_with_dash# ML model dependencies

from dash_plotter import DashRealTimePlottertry:

    import tensorflow as tf

class SimpleZabbixConnector:    from sklearn.preprocessing import StandardScaler

    """Simple Zabbix data connector for anomaly detection"""except ImportError:

        print("❌ ML dependencies not installed. Install with: pip install tensorflow scikit-learn")

    def __init__(self, config_path: str = "config.json"):    sys.exit(1)

        self.config = self._load_config(config_path)

        self.zabbix_api = None# Import your existing modules (adjust paths as needed)

        self.logger = self._setup_logging()sys.path.append('/opt/anomaly_detection')  # Production path

        sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Local development

        # Connect to Zabbix

        self._connect_zabbix()try:

            from data_utils import *

        # Settings    from initial_model import get_initial_model

        self.context_length = 60    from online_forecasting_multi_step import multistep_rolling_buffer_learning_prediction_with_dash

        self.update_interval = self.config.get('monitoring', {}).get('update_interval', 60)    from dash_plotter import DashRealTimePlotter

            import pickle

        # Variable mappings from your training to Zabbixexcept ImportError as e:

        self.variable_mappings = {    print(f"⚠️  Warning: Could not import existing modules: {e}")

            'ICMP response time': ['icmpping', 'icmppingsec'],    print("   Make sure your project files are accessible")

            'Switch 1 - Temperature': ['sensor.temp.1', 'temp.1', 'temperature.1'],    

            'Switch 2 - Temperature': ['sensor.temp.2', 'temp.2', 'temperature.2'],    # Fallback imports for essential functionality

            'CPU utilization': ['system.cpu.util', 'cpu.util'],    try:

            'Memory utilization': ['vm.memory.util', 'memory.util'],        import pickle

            'Interface Bits received': ['net.if.in'],    except ImportError:

            'Interface Bits sent': ['net.if.out']        print("❌ Critical: pickle module not available")

        }        sys.exit(1)

        

        self.logger.info("🏭 Simple Zabbix Connector initialized")class ZabbixDataConnector:

    """

    def _load_config(self, config_path: str) -> Dict:    Zabbix Data Connector - READ-ONLY mode for industrial anomaly detection

        """Load configuration"""    

        try:    Fetches data from Zabbix and provides it in the format expected by

            with open(config_path, 'r') as f:    your existing online forecasting models.

                return json.load(f)    """

        except Exception as e:    

            print(f"❌ Config error: {e}")    def __init__(self, config_path: str = "/etc/zabbix/ml_config.json"):

            sys.exit(1)        """Initialize the Zabbix data connector"""

        self.config = self._load_config(config_path)

    def _setup_logging(self) -> logging.Logger:        self.zabbix_api = None

        """Setup simple logging"""        self.logger = self._setup_logging()

        logger = logging.getLogger('ZabbixConnector')        self.cache_db_path = self.config.get('database', {}).get('path', '/tmp/ml_cache.db')

        logger.setLevel(logging.INFO)        

                # Initialize database

        handler = logging.StreamHandler()        self._init_database()

        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))        

        logger.addHandler(handler)        # Connect to Zabbix

                self._connect_zabbix()

        return logger        

        # Industrial host discovery

    def _connect_zabbix(self):        self.industrial_hosts = {}

        """Connect to Zabbix API"""        self.metric_mappings = {}

        try:        

            zabbix_config = self.config['zabbix']        # Data collection settings

            self.zabbix_api = ZabbixAPI(zabbix_config['url'])        self.context_length = self.config.get('monitoring', {}).get('context_length', 60)

            self.zabbix_api.login(zabbix_config['user'], zabbix_config['password'])        self.update_interval = self.config.get('monitoring', {}).get('update_interval', 60)

                    

            api_info = self.zabbix_api.apiinfo.version()        # Variable mapping from your training data to Zabbix metrics

            self.logger.info(f"✅ Connected to Zabbix {api_info}")        self.variable_mappings = {

                        'ICMP response time': ['icmpping', 'icmppingsec'],

        except Exception as e:            'Switch 1 - Temperature': ['sensor.temp.1', 'temp.1', 'temperature.1'],

            self.logger.error(f"❌ Zabbix connection failed: {e}")            'Switch 2 - Temperature': ['sensor.temp.2', 'temp.2', 'temperature.2'], 

            raise            '#11: CPU utilization': ['system.cpu.util', 'cpu.util'],

            'Processor: Used memory': ['vm.memory.util', 'memory.util'],

    def get_industrial_hosts(self) -> List[Dict]:            'Interface Gi0/0/0: Bits received': ['net.if.in[Gi0/0/0]', 'net.if.in'],

        """Get industrial hosts from configured groups"""            'Interface Gi0/0/0: Bits sent': ['net.if.out[Gi0/0/0]', 'net.if.out'],

        try:            'Interface Gi0/0/1: Bits received': ['net.if.in[Gi0/0/1]'],

            device_groups = self.config.get('industrial_filters', {}).get('device_groups', ['Industrial'])            'Interface Gi0/0/1: Bits sent': ['net.if.out[Gi0/0/1]'],

                        # Add more mappings as needed

            # Get groups        }

            groups = self.zabbix_api.hostgroup.get(        

                filter={'name': device_groups},        self.logger.info("🏭 Zabbix Data Connector initialized (READ-ONLY mode)")

                output=['groupid', 'name']

            )    def _load_config(self, config_path: str) -> Dict:

                    """Load configuration from JSON file"""

            if not groups:        try:

                self.logger.warning(f"No industrial groups found: {device_groups}")            with open(config_path, 'r') as f:

                return []                config = json.load(f)

                        return config

            group_ids = [group['groupid'] for group in groups]        except FileNotFoundError:

                        print(f"❌ Configuration file not found: {config_path}")

            # Get hosts            sys.exit(1)

            hosts = self.zabbix_api.host.get(        except json.JSONDecodeError as e:

                groupids=group_ids,            print(f"❌ Invalid JSON in configuration file: {e}")

                output=['hostid', 'host', 'name'],            sys.exit(1)

                filter={'status': 0}

            )    def _setup_logging(self) -> logging.Logger:

                    """Setup logging configuration"""

            self.logger.info(f"Found {len(hosts)} industrial hosts")        logging_config = self.config.get('logging', {})

            return hosts        log_file = logging_config.get('file', '/var/log/zabbix/ml_anomaly.log')

                    log_level = logging_config.get('level', 'INFO')

        except Exception as e:        

            self.logger.error(f"Host discovery failed: {e}")        # Create log directory if it doesn't exist

            return []        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        

    def fetch_host_data(self, host_id: str, hours_back: int = 2) -> pd.DataFrame:        # Configure logger

        """Fetch data for a host"""        logger = logging.getLogger('ZabbixDataConnector')

        try:        logger.setLevel(getattr(logging, log_level))

            # Time range        

            time_till = int(time.time())        # File handler

            time_from = time_till - (hours_back * 3600)        file_handler = logging.FileHandler(log_file)

                    file_handler.setFormatter(

            # Get all items for host            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            items = self.zabbix_api.item.get(        )

                hostids=[host_id],        logger.addHandler(file_handler)

                output=['itemid', 'key_', 'name'],        

                filter={'status': 0}        # Console handler

            )        console_handler = logging.StreamHandler()

                    console_handler.setFormatter(

            if not items:            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

                return pd.DataFrame()        )

                    logger.addHandler(console_handler)

            # Map items to variables        

            item_mappings = {}        return logger

            for item in items:

                item_key = item['key_']    def _init_database(self):

                for variable_name, possible_keys in self.variable_mappings.items():        """Initialize SQLite database for data caching"""

                    for possible_key in possible_keys:        try:

                        if possible_key in item_key:            os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)

                            item_mappings[item['itemid']] = variable_name            

                            break            with sqlite3.connect(self.cache_db_path) as conn:

                            cursor = conn.cursor()

            if not item_mappings:                

                return pd.DataFrame()                # Create metrics table

                            cursor.execute('''

            # Fetch history                    CREATE TABLE IF NOT EXISTS metrics (

            history = self.zabbix_api.history.get(                        id INTEGER PRIMARY KEY AUTOINCREMENT,

                itemids=list(item_mappings.keys()),                        timestamp INTEGER NOT NULL,

                time_from=time_from,                        host_id TEXT NOT NULL,

                time_till=time_till,                        host_name TEXT NOT NULL,

                output=['itemid', 'clock', 'value'],                        variable_name TEXT NOT NULL,

                sortfield='clock'                        value REAL NOT NULL,

            )                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

                                )

            if not history:                ''')

                return pd.DataFrame()                

                            # Create index for faster queries

            # Process into DataFrame                cursor.execute('''

            data_records = []                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp_host 

            for record in history:                    ON metrics(timestamp, host_id, variable_name)

                if record['itemid'] in item_mappings:                ''')

                    data_records.append({                

                        'timestamp': int(record['clock']),                conn.commit()

                        'variable': item_mappings[record['itemid']],                

                        'value': float(record['value'])            self.logger.info(f"✅ Database initialized: {self.cache_db_path}")

                    })            

                    except Exception as e:

            if not data_records:            self.logger.error(f"❌ Database initialization failed: {e}")

                return pd.DataFrame()            raise

            

            # Create DataFrame    def _connect_zabbix(self):

            df = pd.DataFrame(data_records)        """Connect to Zabbix API"""

            df_pivot = df.pivot_table(        try:

                index='timestamp',            zabbix_config = self.config['zabbix']

                columns='variable',             self.zabbix_api = ZabbixAPI(zabbix_config['url'])

                values='value',            self.zabbix_api.login(zabbix_config['user'], zabbix_config['password'])

                aggfunc='first'            

            )            # Test connection

                        api_info = self.zabbix_api.apiinfo.version()

            # Convert to datetime and resample            self.logger.info(f"✅ Connected to Zabbix API version: {api_info}")

            df_pivot.index = pd.to_datetime(df_pivot.index, unit='s')            

            df_resampled = df_pivot.resample('1T').mean()        except Exception as e:

            df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')            self.logger.error(f"❌ Failed to connect to Zabbix: {e}")

                        raise

            return df_resampled

                def discover_industrial_hosts(self) -> Dict[str, Dict]:

        except Exception as e:        """

            self.logger.error(f"Data fetch failed for host {host_id}: {e}")        Discover industrial hosts from Zabbix based on configured groups

            return pd.DataFrame()        Returns: Dict[host_id, host_info]

        """

    def get_online_data(self) -> Tuple[pd.DataFrame, Dict, int, List[str]]:        try:

        """Get data in format expected by your models"""            industrial_filters = self.config.get('industrial_filters', {})

        try:            device_groups = industrial_filters.get('device_groups', ['Industrial'])

            self.logger.info("🔄 Fetching data from Zabbix...")            

                        # Get group IDs

            # Get hosts            groups = self.zabbix_api.hostgroup.get(

            hosts = self.get_industrial_hosts()                filter={'name': device_groups},

            if not hosts:                output=['groupid', 'name']

                raise ValueError("No industrial hosts found")            )

                        

            # Collect data from all hosts            if not groups:

            all_data = []                self.logger.warning(f"⚠️  No industrial groups found: {device_groups}")

            for host in hosts:                return {}

                self.logger.info(f"📊 Getting data from {host['name']}")            

                host_data = self.fetch_host_data(host['hostid'])            group_ids = [group['groupid'] for group in groups]

                            self.logger.info(f"📍 Found {len(groups)} industrial groups")

                if not host_data.empty:            

                    # Add host prefix to columns            # Get hosts in these groups

                    host_data.columns = [f"{host['host']}_{col}" for col in host_data.columns]            hosts = self.zabbix_api.host.get(

                    all_data.append(host_data)                groupids=group_ids,

                            output=['hostid', 'host', 'name', 'status'],

            if not all_data:                filter={'status': 0}  # Only enabled hosts

                raise ValueError("No data retrieved")            )

                        

            # Combine all host data            industrial_hosts = {}

            df_combined = pd.concat(all_data, axis=1, sort=True)            for host in hosts:

            df_resampled = df_combined.resample('1T').mean()                host_id = host['hostid']

            df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')                industrial_hosts[host_id] = {

                                'hostid': host_id,

            variables = list(df_resampled.columns)                    'hostname': host['host'],

                                'display_name': host['name'],

            # Load or create scalers                    'status': host['status']

            model_path = self.config.get('models', {}).get('forecasting_model_path')                }

            scalers_file = os.path.join(model_path, 'scalers_train.pkl') if model_path else None            

                        self.logger.info(f"🏭 Discovered {len(industrial_hosts)} industrial hosts")

            if scalers_file and os.path.exists(scalers_file):            return industrial_hosts

                with open(scalers_file, 'rb') as f:            

                    scalers = pickle.load(f)        except Exception as e:

                self.logger.info("✅ Loaded model scalers")            self.logger.error(f"❌ Host discovery failed: {e}")

            else:            return {}

                # Create new scalers

                scalers = {}    def map_host_metrics(self, host_id: str) -> Dict[str, List[str]]:

                for var in variables:        """

                    scaler = StandardScaler()        Map training variables to actual Zabbix items for a specific host

                    scaler.fit(df_resampled[var].values.reshape(-1, 1))        Returns: Dict[variable_name, [item_keys]]

                    scalers[var] = scaler        """

                self.logger.warning("⚠️ Created new scalers")        try:

                        # Get all items for this host

            self.logger.info(f"✅ Retrieved {len(df_resampled)} points, {len(variables)} variables")            items = self.zabbix_api.item.get(

            return df_resampled, scalers, self.context_length, variables                hostids=[host_id],

                            output=['itemid', 'key_', 'name', 'value_type', 'status'],

        except Exception as e:                filter={'status': 0}  # Only enabled items

            self.logger.error(f"❌ Data retrieval failed: {e}")            )

            raise            

            # Create mapping

    def run_monitoring(self):            host_mappings = {}

        """Main monitoring loop"""            available_keys = [item['key_'] for item in items]

        try:            

            self.logger.info("🚀 Starting monitoring loop...")            for variable_name, possible_keys in self.variable_mappings.items():

                            matched_keys = []

            # Load your trained model                for possible_key in possible_keys:

            model_path = self.config.get('models', {}).get('forecasting_model_path')                    # Find matching items (exact match or pattern match)

            if not model_path or not os.path.exists(model_path):                    for item_key in available_keys:

                raise FileNotFoundError(f"Model not found: {model_path}")                        if possible_key in item_key or item_key.startswith(possible_key.split('[')[0]):

                                        matched_keys.append(item_key)

            initial_model = get_initial_model(model_path)                

                            if matched_keys:

            # Start dashboard                    host_mappings[variable_name] = list(set(matched_keys))  # Remove duplicates

            plotter = DashRealTimePlotter()            

            plotter.start_server()            self.logger.info(f"📊 Mapped {len(host_mappings)} variables for host {host_id}")

            print("🌐 Dashboard: http://localhost:8050")            return host_mappings

            time.sleep(3)            

                    except Exception as e:

            # Main loop            self.logger.error(f"❌ Metric mapping failed for host {host_id}: {e}")

            while True:            return {}

                try:

                    self.logger.info("🔄 Fetching fresh data...")    def fetch_host_data(self, host_id: str, time_from: int, time_till: int) -> pd.DataFrame:

                            """

                    # Get data (this replaces your df_online)        Fetch historical data for a specific host

                    df_online, scalers, context_length, variables = self.get_online_data()        Returns: DataFrame with timestamp index and variable columns

                            """

                    if df_online.empty:        try:

                        self.logger.warning("No data, waiting...")            if host_id not in self.metric_mappings:

                        time.sleep(self.update_interval)                self.metric_mappings[host_id] = self.map_host_metrics(host_id)

                        continue            

                                mappings = self.metric_mappings[host_id]

                    # Prepare for your forecasting function            if not mappings:

                    df_removed_nans_forecasting = df_online.copy()                self.logger.warning(f"⚠️  No metric mappings for host {host_id}")

                    df_removed_nans_classification = df_online.copy()                return pd.DataFrame()

                    prediction_horizon = 6            

                                # Collect all item IDs

                    self.logger.info(f"🎯 Running forecasting...")            all_item_keys = []

                                for variable_name, item_keys in mappings.items():

                    # Run your existing forecasting                all_item_keys.extend(item_keys)

                    predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df = \            

                        multistep_rolling_buffer_learning_prediction_with_dash(            # Get item IDs

                            initial_model=initial_model,            items = self.zabbix_api.item.get(

                            df_online=df_online,                hostids=[host_id],

                            scalers=scalers,                filter={'key_': all_item_keys},

                            context_length=context_length,                output=['itemid', 'key_', 'name']

                            df_removed_nans_forecasting=df_removed_nans_forecasting,            )

                            df_removed_nans_classification=df_removed_nans_classification,            

                            dash_plotter=plotter,            item_id_to_key = {item['itemid']: item['key_'] for item in items}

                            variables=variables,            key_to_variable = {}

                            prediction_horizon=prediction_horizon            

                        )            # Create reverse mapping (item_key -> variable_name)

                                for variable_name, item_keys in mappings.items():

                    self.logger.info("✅ Forecasting completed")                for item_key in item_keys:

                                        key_to_variable[item_key] = variable_name

                    # Wait for next cycle            

                    time.sleep(self.update_interval)            if not item_id_to_key:

                                    self.logger.warning(f"⚠️  No items found for host {host_id}")

                except KeyboardInterrupt:                return pd.DataFrame()

                    self.logger.info("🛑 Shutdown requested")            

                    break            # Fetch historical data

                except Exception as e:            history = self.zabbix_api.history.get(

                    self.logger.error(f"❌ Loop error: {e}")                itemids=list(item_id_to_key.keys()),

                    time.sleep(self.update_interval)                time_from=time_from,

                            time_till=time_till,

        except Exception as e:                output=['itemid', 'clock', 'value'],

            self.logger.error(f"❌ Fatal error: {e}")                sortfield='clock',

            raise                sortorder='ASC'

            )

def main():            

    """Main entry point"""            if not history:

    import argparse                self.logger.warning(f"⚠️  No historical data for host {host_id}")

                    return pd.DataFrame()

    parser = argparse.ArgumentParser(description='Simple Zabbix Industrial Monitoring')            

    parser.add_argument('--config', '-c', default='config.json', help='Config file')            # Process data into DataFrame

    parser.add_argument('--test', '-t', action='store_true', help='Test mode only')            data_records = []

                for record in history:

    args = parser.parse_args()                item_id = record['itemid']

                    if item_id in item_id_to_key:

    try:                    item_key = item_id_to_key[item_id]

        connector = SimpleZabbixConnector(args.config)                    if item_key in key_to_variable:

                                variable_name = key_to_variable[item_key]

        if args.test:                        timestamp = int(record['clock'])

            print("🧪 Testing connection...")                        value = float(record['value'])

            df_online, scalers, context_length, variables = connector.get_online_data()                        

            print(f"✅ Success! Shape: {df_online.shape}, Variables: {len(variables)}")                        data_records.append({

        else:                            'timestamp': timestamp,

            print("🏭 Starting production monitoring...")                            'variable': variable_name,

            connector.run_monitoring()                            'value': value

                                    })

    except KeyboardInterrupt:            

        print("\n🛑 Shutdown")            if not data_records:

    except Exception as e:                return pd.DataFrame()

        print(f"❌ Error: {e}")            

        sys.exit(1)            # Create DataFrame

            df = pd.DataFrame(data_records)

if __name__ == "__main__":            

    main()            # Pivot to get variables as columns
            df_pivot = df.pivot_table(
                index='timestamp', 
                columns='variable', 
                values='value', 
                aggfunc='first'
            )
            
            # Convert timestamp to datetime index
            df_pivot.index = pd.to_datetime(df_pivot.index, unit='s')
            
            # Fill missing values with forward fill, then backward fill
            df_pivot = df_pivot.fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info(f"📈 Fetched {len(df_pivot)} data points for host {host_id}")
            return df_pivot
            
        except Exception as e:
            self.logger.error(f"❌ Data fetch failed for host {host_id}: {e}")
            return pd.DataFrame()

    def get_online_data_from_zabbix(self, hours_back: int = 2) -> Tuple[pd.DataFrame, Dict, int, List[str]]:
        """
        Get recent data from Zabbix in the format expected by your online forecasting models
        
        Returns:
            df_online: DataFrame with industrial data (replaces your df_online)
            scalers_train: Dictionary of scalers (loaded from your model)
            context_length: Context length for the model
            variables: List of variable names
        """
        try:
            # Discover hosts if not already done
            if not self.industrial_hosts:
                self.industrial_hosts = self.discover_industrial_hosts()
            
            if not self.industrial_hosts:
                raise ValueError("No industrial hosts found")
            
            # Calculate time range
            time_till = int(time.time())
            time_from = time_till - (hours_back * 3600)
            
            self.logger.info(f"🕒 Fetching {hours_back} hours of data from Zabbix...")
            
            # Collect data from all hosts
            all_host_data = []
            
            for host_id, host_info in self.industrial_hosts.items():
                self.logger.info(f"📊 Fetching data for {host_info['display_name']}...")
                
                host_data = self.fetch_host_data(host_id, time_from, time_till)
                if not host_data.empty:
                    # Add host identifier to columns
                    host_data.columns = [f"{host_info['hostname']}_{col}" for col in host_data.columns]
                    all_host_data.append(host_data)
            
            if not all_host_data:
                raise ValueError("No data retrieved from any hosts")
            
            # Combine all host data
            df_combined = pd.concat(all_host_data, axis=1, sort=True)
            
            # Fill missing values and resample to 1-minute intervals
            df_resampled = df_combined.resample('1T').mean()
            df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')
            
            # Get variables list (column names)
            variables = list(df_resampled.columns)
            
            # Load scalers from your trained model
            models_config = self.config.get('models', {})
            forecasting_model_path = models_config.get('forecasting_model_path')
            
            if forecasting_model_path and os.path.exists(os.path.join(forecasting_model_path, 'scalers_train.pkl')):
                with open(os.path.join(forecasting_model_path, 'scalers_train.pkl'), 'rb') as f:
                    scalers_train = pickle.load(f)
                self.logger.info("✅ Loaded scalers from trained model")
            else:
                # Create new scalers if model scalers not available
                scalers_train = {}
                for var in variables:
                    scaler = StandardScaler()
                    scaler.fit(df_resampled[var].values.reshape(-1, 1))
                    scalers_train[var] = scaler
                self.logger.warning("⚠️  Created new scalers (model scalers not found)")
            
            # Cache data to database
            self._cache_data_to_db(df_resampled)
            
            self.logger.info(f"✅ Retrieved {len(df_resampled)} data points with {len(variables)} variables")
            
            return df_resampled, scalers_train, self.context_length, variables
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get online data from Zabbix: {e}")
            raise

    def _cache_data_to_db(self, df: pd.DataFrame):
        """Cache DataFrame data to database"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                
                # Clear old data (keep last 24 hours)
                cutoff_time = int(time.time()) - (24 * 3600)
                cursor.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff_time,))
                
                # Insert new data
                for timestamp, row in df.iterrows():
                    timestamp_unix = int(timestamp.timestamp())
                    
                    for variable_name, value in row.items():
                        if pd.notna(value):
                            # Extract host name from variable name (assumes format: hostname_variable)
                            if '_' in variable_name:
                                host_name = variable_name.split('_')[0]
                                clean_variable = '_'.join(variable_name.split('_')[1:])
                            else:
                                host_name = 'unknown'
                                clean_variable = variable_name
                            
                            cursor.execute('''
                                INSERT OR REPLACE INTO metrics 
                                (timestamp, host_id, host_name, variable_name, value)
                                VALUES (?, ?, ?, ?, ?)
                            ''', (timestamp_unix, host_name, host_name, clean_variable, float(value)))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to cache data: {e}")

    def run_online_forecasting_loop(self):
        """
        Main loop that integrates with your existing online forecasting code
        This replaces the data loading part of your main.py
        """
        try:
            self.logger.info("🚀 Starting online forecasting loop with Zabbix data...")
            
            # Load your trained models
            models_config = self.config.get('models', {})
            initial_model_path = models_config.get('forecasting_model_path')
            
            if not initial_model_path or not os.path.exists(initial_model_path):
                raise FileNotFoundError(f"Forecasting model not found: {initial_model_path}")
            
            # Load initial model
            self.logger.info("📦 Loading trained forecasting model...")
            initial_model = get_initial_model(initial_model_path)
            
            # Initialize Dash plotter
            plotter = DashRealTimePlotter()
            self.logger.info("🎯 Starting Dash server...")
            plotter.start_server()
            
            print("🌐 Open http://localhost:8050 in your browser to view real-time plots")
            time.sleep(3)  # Allow server to initialize
            
            # Main monitoring loop
            while True:
                try:
                    self.logger.info("🔄 Fetching fresh data from Zabbix...")
                    
                    # Get online data from Zabbix (this replaces your df_online)
                    df_online, scalers_train, context_length, variables = self.get_online_data_from_zabbix(hours_back=2)
                    
                    if df_online.empty:
                        self.logger.warning("⚠️  No data received, waiting for next cycle...")
                        time.sleep(self.update_interval)
                        continue
                    
                    # Prepare additional data structures (using cached data or defaults)
                    df_removed_nans_forecasting = df_online.copy()
                    df_removed_nans_classification = df_online.copy()
                    
                    prediction_horizon = self.config.get('monitoring', {}).get('prediction_horizon', 6)
                    
                    self.logger.info(f"🎯 Running forecasting with {len(df_online)} data points...")
                    
                    # Run your existing multi-step forecasting
                    predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df = \
                        multistep_rolling_buffer_learning_prediction_with_dash(
                            initial_model=initial_model,
                            df_online=df_online,
                            scalers=scalers_train,
                            context_length=context_length,
                            df_removed_nans_forecasting=df_removed_nans_forecasting,
                            df_removed_nans_classification=df_removed_nans_classification,
                            dash_plotter=plotter,
                            variables=variables,
                            prediction_horizon=prediction_horizon
                        )
                    
                    self.logger.info("✅ Forecasting cycle completed successfully")
                    
                    # Wait for next cycle
                    self.logger.info(f"⏱️  Waiting {self.update_interval} seconds for next cycle...")
                    time.sleep(self.update_interval)
                    
                except KeyboardInterrupt:
                    self.logger.info("🛑 Keyboard interrupt received, shutting down...")
                    break
                except Exception as e:
                    self.logger.error(f"❌ Error in forecasting loop: {e}")
                    self.logger.info(f"⏱️  Waiting {self.update_interval} seconds before retry...")
                    time.sleep(self.update_interval)
            
        except Exception as e:
            self.logger.error(f"❌ Fatal error in online forecasting loop: {e}")
            raise

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Zabbix Data Connector for Industrial Anomaly Detection')
    parser.add_argument('--config', '-c', default='/etc/zabbix/ml_config.json',
                       help='Configuration file path')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Test connection and data fetch only')
    
    args = parser.parse_args()
    
    try:
        # Initialize connector
        connector = ZabbixDataConnector(args.config)
        
        if args.test:
            # Test mode - just fetch data and display info
            print("🧪 Running in test mode...")
            
            df_online, scalers_train, context_length, variables = connector.get_online_data_from_zabbix()
            
            print(f"✅ Test successful!")
            print(f"   Data shape: {df_online.shape}")
            print(f"   Variables: {variables}")
            print(f"   Context length: {context_length}")
            print(f"   Time range: {df_online.index.min()} to {df_online.index.max()}")
            
        else:
            # Production mode - run continuous monitoring
            print("🏭 Starting production monitoring...")
            connector.run_online_forecasting_loop()
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()