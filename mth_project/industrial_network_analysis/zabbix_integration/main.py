#!/usr/bin/env python3#!/usr/bin/env python3

""""""

🏭 Zabbix Data Connector for Industrial Anomaly Detection🏭 Zabbix Data Connector for Industrial Anomaly Detection

READ-ONLY integration that fetches data from Zabbix and runs ML modelsREAD-ONLY integration that fetches data from Zabbix and runs ML models

""""""



import sysimport sys

import osimport os

import jsonimport json

import timeimport time

import loggingimport logging

import pickleimport pickle

from typing import Dict, List, Tuplefrom typing import Dict, List, Tuple



# Core dependencies# Core dependencies

import numpy as npimport numpy as np

import pandas as pdimport pandas as pd



# Zabbix API# Zabbix API

try:try:

    from pyzabbix import ZabbixAPI    from pyzabbix import ZabbixAPI

except ImportError:except ImportError:

    print("❌ Install pyzabbix: pip install pyzabbix")    print("❌ Install pyzabbix: pip install pyzabbix")

    sys.exit(1)    sys.exit(1)



# ML dependencies# ML dependencies

try:try:

    import tensorflow as tf    import tensorflow as tf

    from sklearn.preprocessing import StandardScaler    from sklearn.preprocessing import StandardScaler

except ImportError:except ImportError:

    print("❌ Install ML dependencies: pip install tensorflow scikit-learn")    print("❌ Install ML dependencies: pip install tensorflow scikit-learn")

    sys.exit(1)    sys.exit(1)



# Import existing modules# Import existing modules

sys.path.append('/opt/anomaly_detection')sys.path.append('/opt/anomaly_detection')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))sys.path.append(os.path.dirname(os.path.abspath(__file__)))



try:try:

    from data_utils import *    from data_utils import *

    from initial_model import get_initial_model    from initial_model import get_initial_model

    from online_forecasting_multi_step import multistep_rolling_buffer_learning_prediction_with_dash    from online_forecasting_multi_step import multistep_rolling_buffer_learning_prediction_with_dash

    from dash_plotter import DashRealTimePlotter    from dash_plotter import DashRealTimePlotter

except ImportError as e:except ImportError as e:

    print(f"⚠️  Warning: Could not import project modules: {e}")    print(f"⚠️  Warning: Could not import project modules: {e}")



class ZabbixDataConnector:class ZabbixDataConnector:

    """Simple Zabbix data connector for industrial anomaly detection"""    """Simple Zabbix data connector for industrial anomaly detection"""

        

    def __init__(self, config_path: str = "config.json"):    def __init__(self, config_path: str = "config.json"):

        """Initialize the connector"""        """Initialize the connector"""

        self.config = self._load_config(config_path)        self.config = self._load_config(config_path)

        self.zabbix_api = None        self.zabbix_api = None

        self.logger = self._setup_logging()        self.logger = self._setup_logging()

                

        # Connect to Zabbix        # Connect to Zabbix

        self._connect_zabbix()        self._connect_zabbix()

                

        # Settings        # Settings

        self.context_length = self.config.get('monitoring', {}).get('context_length', 60)        self.context_length = self.config.get('monitoring', {}).get('context_length', 60)

        self.update_interval = self.config.get('monitoring', {}).get('update_interval', 60)        self.update_interval = self.config.get('monitoring', {}).get('update_interval', 60)

                

        # Variable mappings from your training data to Zabbix metrics        # Variable mappings from your training data to Zabbix metrics

        self.variable_mappings = {        self.variable_mappings = {

            'ICMP response time': ['icmpping', 'icmppingsec'],            'ICMP response time': ['icmpping', 'icmppingsec'],

            'Switch 1 - Temperature': ['sensor.temp.1', 'temp.1', 'temperature.1'],            'Switch 1 - Temperature': ['sensor.temp.1', 'temp.1', 'temperature.1'],

            'Switch 2 - Temperature': ['sensor.temp.2', 'temp.2', 'temperature.2'],            'Switch 2 - Temperature': ['sensor.temp.2', 'temp.2', 'temperature.2'],

            'CPU utilization': ['system.cpu.util', 'cpu.util'],            'CPU utilization': ['system.cpu.util', 'cpu.util'],

            'Memory utilization': ['vm.memory.util', 'memory.util'],            'Memory utilization': ['vm.memory.util', 'memory.util'],

            'Interface Bits received': ['net.if.in'],            'Interface Bits received': ['net.if.in'],

            'Interface Bits sent': ['net.if.out']            'Interface Bits sent': ['net.if.out']

        }        }

                

        self.logger.info("🏭 Zabbix Data Connector initialized")        self.logger.info("🏭 Zabbix Data Connector initialized")



    def _load_config(self, config_path: str) -> Dict:    def _load_config(self, config_path: str) -> Dict:

        """Load configuration from JSON file"""        """Load configuration from JSON file"""

        try:        try:

            with open(config_path, 'r') as f:            with open(config_path, 'r') as f:

                return json.load(f)                return json.load(f)

        except FileNotFoundError:        except FileNotFoundError:

            print(f"❌ Configuration file not found: {config_path}")            print(f"❌ Configuration file not found: {config_path}")

            sys.exit(1)            sys.exit(1)

        except json.JSONDecodeError as e:        except json.JSONDecodeError as e:

            print(f"❌ Invalid JSON in configuration file: {e}")            print(f"❌ Invalid JSON in configuration file: {e}")

            sys.exit(1)            sys.exit(1)



    def _setup_logging(self) -> logging.Logger:    def _setup_logging(self) -> logging.Logger:

        """Setup logging"""        """Setup logging"""

        logger = logging.getLogger('ZabbixDataConnector')        logger = logging.getLogger('ZabbixDataConnector')

        logger.setLevel(logging.INFO)        logger.setLevel(logging.INFO)

                

        # Console handler        # Console handler

        handler = logging.StreamHandler()        handler = logging.StreamHandler()

        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        logger.addHandler(handler)        logger.addHandler(handler)

                

        return logger        return logger



    def _connect_zabbix(self):    def _connect_zabbix(self):

        """Connect to Zabbix API"""        """Connect to Zabbix API"""

        try:        try:

            zabbix_config = self.config['zabbix']            zabbix_config = self.config['zabbix']

            self.zabbix_api = ZabbixAPI(zabbix_config['url'])            self.zabbix_api = ZabbixAPI(zabbix_config['url'])

            self.zabbix_api.login(zabbix_config['user'], zabbix_config['password'])            self.zabbix_api.login(zabbix_config['user'], zabbix_config['password'])

                        

            # Test connection            # Test connection

            api_info = self.zabbix_api.apiinfo.version()            api_info = self.zabbix_api.apiinfo.version()

            self.logger.info(f"✅ Connected to Zabbix API version: {api_info}")            self.logger.info(f"✅ Connected to Zabbix API version: {api_info}")

                        

        except Exception as e:        except Exception as e:

            self.logger.error(f"❌ Failed to connect to Zabbix: {e}")            self.logger.error(f"❌ Failed to connect to Zabbix: {e}")

            raise            raise



    def get_industrial_hosts(self) -> List[Dict]:    def get_industrial_hosts(self) -> List[Dict]:

        """Get industrial hosts from configured groups"""        """Get industrial hosts from configured groups"""

        try:        try:

            device_groups = self.config.get('industrial_filters', {}).get('device_groups', ['Industrial'])            device_groups = self.config.get('industrial_filters', {}).get('device_groups', ['Industrial'])

                        

            # Get groups            # Get groups

            groups = self.zabbix_api.hostgroup.get(            groups = self.zabbix_api.hostgroup.get(

                filter={'name': device_groups},                filter={'name': device_groups},

                output=['groupid', 'name']                output=['groupid', 'name']

            )            )

                        

            if not groups:            if not groups:

                self.logger.warning(f"⚠️  No industrial groups found: {device_groups}")                self.logger.warning(f"⚠️  No industrial groups found: {device_groups}")

                return []                return []

                        

            group_ids = [group['groupid'] for group in groups]            group_ids = [group['groupid'] for group in groups]

                        

            # Get hosts in these groups            # Get hosts in these groups

            hosts = self.zabbix_api.host.get(            hosts = self.zabbix_api.host.get(

                groupids=group_ids,                groupids=group_ids,

                output=['hostid', 'host', 'name'],                output=['hostid', 'host', 'name'],

                filter={'status': 0}  # Only enabled hosts                filter={'status': 0}  # Only enabled hosts

            )            )

                        

            self.logger.info(f"🏭 Found {len(hosts)} industrial hosts")            self.logger.info(f"🏭 Found {len(hosts)} industrial hosts")

            return hosts            return hosts

                        

        except Exception as e:        except Exception as e:

            self.logger.error(f"❌ Host discovery failed: {e}")            self.logger.error(f"❌ Host discovery failed: {e}")

            return []            return []



    def fetch_host_data(self, host_id: str, hours_back: int = 2) -> pd.DataFrame:    def fetch_host_data(self, host_id: str, hours_back: int = 2) -> pd.DataFrame:

        """Fetch historical data for a specific host"""        """Fetch historical data for a specific host"""

        try:        try:

            # Calculate time range            # Calculate time range

            time_till = int(time.time())            time_till = int(time.time())

            time_from = time_till - (hours_back * 3600)            time_from = time_till - (hours_back * 3600)

                        

            # Get all items for this host            # Get all items for this host

            items = self.zabbix_api.item.get(            items = self.zabbix_api.item.get(

                hostids=[host_id],                hostids=[host_id],

                output=['itemid', 'key_', 'name'],                output=['itemid', 'key_', 'name'],

                filter={'status': 0}  # Only enabled items                filter={'status': 0}  # Only enabled items

            )            )

                        

            if not items:            if not items:

                self.logger.warning(f"⚠️  No items found for host {host_id}")                self.logger.warning(f"⚠️  No items found for host {host_id}")

                return pd.DataFrame()                return pd.DataFrame()

                        

            # Map items to variables            # Map items to variables

            item_mappings = {}            item_mappings = {}

            for item in items:            for item in items:

                item_key = item['key_']                item_key = item['key_']

                for variable_name, possible_keys in self.variable_mappings.items():                for variable_name, possible_keys in self.variable_mappings.items():

                    for possible_key in possible_keys:                    for possible_key in possible_keys:

                        if possible_key in item_key:                        if possible_key in item_key:

                            item_mappings[item['itemid']] = variable_name                            item_mappings[item['itemid']] = variable_name

                            break                            break

                        

            if not item_mappings:            if not item_mappings:

                self.logger.warning(f"⚠️  No metric mappings for host {host_id}")                self.logger.warning(f"⚠️  No metric mappings for host {host_id}")

                return pd.DataFrame()                return pd.DataFrame()

                        

            # Fetch historical data            # Fetch historical data

            history = self.zabbix_api.history.get(            history = self.zabbix_api.history.get(

                itemids=list(item_mappings.keys()),                itemids=list(item_mappings.keys()),

                time_from=time_from,                time_from=time_from,

                time_till=time_till,                time_till=time_till,

                output=['itemid', 'clock', 'value'],                output=['itemid', 'clock', 'value'],

                sortfield='clock',                sortfield='clock',

                sortorder='ASC'                sortorder='ASC'

            )            )

                        

            if not history:            if not history:

                self.logger.warning(f"⚠️  No historical data for host {host_id}")                self.logger.warning(f"⚠️  No historical data for host {host_id}")

                return pd.DataFrame()                return pd.DataFrame()

                        

            # Process data into DataFrame            # Process data into DataFrame

            data_records = []            data_records = []

            for record in history:            for record in history:

                item_id = record['itemid']                item_id = record['itemid']

                if item_id in item_mappings:                if item_id in item_mappings:

                    data_records.append({                    data_records.append({

                        'timestamp': int(record['clock']),                        'timestamp': int(record['clock']),

                        'variable': item_mappings[item_id],                        'variable': item_mappings[item_id],

                        'value': float(record['value'])                        'value': float(record['value'])

                    })                    })

                        

            if not data_records:            if not data_records:

                return pd.DataFrame()                return pd.DataFrame()

                        

            # Create DataFrame and pivot            # Create DataFrame and pivot

            df = pd.DataFrame(data_records)            df = pd.DataFrame(data_records)

            df_pivot = df.pivot_table(            df_pivot = df.pivot_table(

                index='timestamp',                 index='timestamp', 

                columns='variable',                 columns='variable', 

                values='value',                 values='value', 

                aggfunc='first'                aggfunc='first'

            )            )

                        

            # Convert timestamp to datetime index            # Convert timestamp to datetime index

            df_pivot.index = pd.to_datetime(df_pivot.index, unit='s')            df_pivot.index = pd.to_datetime(df_pivot.index, unit='s')

                        

            # Fill missing values            # Fill missing values

            df_pivot = df_pivot.fillna(method='ffill').fillna(method='bfill')            df_pivot = df_pivot.fillna(method='ffill').fillna(method='bfill')

                        

            self.logger.info(f"📈 Fetched {len(df_pivot)} data points for host {host_id}")            self.logger.info(f"📈 Fetched {len(df_pivot)} data points for host {host_id}")

            return df_pivot            return df_pivot

                        

        except Exception as e:        except Exception as e:

            self.logger.error(f"❌ Data fetch failed for host {host_id}: {e}")            self.logger.error(f"❌ Data fetch failed for host {host_id}: {e}")

            return pd.DataFrame()            return pd.DataFrame()



    def get_online_data(self) -> Tuple[pd.DataFrame, Dict, int, List[str]]:    def get_online_data(self) -> Tuple[pd.DataFrame, Dict, int, List[str]]:

        """Get data in format expected by your online forecasting models"""        """Get data in format expected by your online forecasting models"""

        try:        try:

            self.logger.info("🔄 Fetching data from Zabbix...")            self.logger.info("🔄 Fetching data from Zabbix...")

                        

            # Get hosts            # Get hosts

            hosts = self.get_industrial_hosts()            hosts = self.get_industrial_hosts()

            if not hosts:            if not hosts:

                raise ValueError("No industrial hosts found")                raise ValueError("No industrial hosts found")

                        

            # Collect data from all hosts            # Collect data from all hosts

            all_host_data = []            all_host_data = []

            for host in hosts:            for host in hosts:

                self.logger.info(f"📊 Fetching data for {host['name']}...")                self.logger.info(f"📊 Fetching data for {host['name']}...")

                host_data = self.fetch_host_data(host['hostid'])                host_data = self.fetch_host_data(host['hostid'])

                                

                if not host_data.empty:                if not host_data.empty:

                    # Add host identifier to columns                    # Add host identifier to columns

                    host_data.columns = [f"{host['host']}_{col}" for col in host_data.columns]                    host_data.columns = [f"{host['host']}_{col}" for col in host_data.columns]

                    all_host_data.append(host_data)                    all_host_data.append(host_data)

                        

            if not all_host_data:            if not all_host_data:

                raise ValueError("No data retrieved from any hosts")                raise ValueError("No data retrieved from any hosts")

                        

            # Combine all host data            # Combine all host data

            df_combined = pd.concat(all_host_data, axis=1, sort=True)            df_combined = pd.concat(all_host_data, axis=1, sort=True)

                        

            # Resample to 1-minute intervals and fill missing values            # Resample to 1-minute intervals and fill missing values

            df_resampled = df_combined.resample('1T').mean()            df_resampled = df_combined.resample('1T').mean()

            df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')            df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')

                        

            # Get variables list            # Get variables list

            variables = list(df_resampled.columns)            variables = list(df_resampled.columns)

                        

            # Load or create scalers            # Load or create scalers

            model_path = self.config.get('models', {}).get('forecasting_model_path')            model_path = self.config.get('models', {}).get('forecasting_model_path')

            scalers_file = os.path.join(model_path, 'scalers_train.pkl') if model_path else None            scalers_file = os.path.join(model_path, 'scalers_train.pkl') if model_path else None

                        

            if scalers_file and os.path.exists(scalers_file):            if scalers_file and os.path.exists(scalers_file):

                with open(scalers_file, 'rb') as f:                with open(scalers_file, 'rb') as f:

                    scalers = pickle.load(f)                    scalers = pickle.load(f)

                self.logger.info("✅ Loaded scalers from trained model")                self.logger.info("✅ Loaded scalers from trained model")

            else:            else:

                # Create new scalers                # Create new scalers

                scalers = {}                scalers = {}

                for var in variables:                for var in variables:

                    scaler = StandardScaler()                    scaler = StandardScaler()

                    scaler.fit(df_resampled[var].values.reshape(-1, 1))                    scaler.fit(df_resampled[var].values.reshape(-1, 1))

                    scalers[var] = scaler                    scalers[var] = scaler

                self.logger.warning("⚠️  Created new scalers (model scalers not found)")                self.logger.warning("⚠️  Created new scalers (model scalers not found)")

                        

            self.logger.info(f"✅ Retrieved {len(df_resampled)} data points with {len(variables)} variables")            self.logger.info(f"✅ Retrieved {len(df_resampled)} data points with {len(variables)} variables")

            return df_resampled, scalers, self.context_length, variables            return df_resampled, scalers, self.context_length, variables

                        

        except Exception as e:        except Exception as e:

            self.logger.error(f"❌ Failed to get online data from Zabbix: {e}")            self.logger.error(f"❌ Failed to get online data from Zabbix: {e}")

            raise            raise



    def run_monitoring_loop(self):    def run_monitoring_loop(self):

        """Main monitoring loop that integrates with your existing forecasting"""        """Main monitoring loop that integrates with your existing forecasting"""

        try:        try:

            self.logger.info("🚀 Starting monitoring loop with Zabbix data...")            self.logger.info("🚀 Starting monitoring loop with Zabbix data...")

                        

            # Load trained model            # Load trained model

            model_path = self.config.get('models', {}).get('forecasting_model_path')            model_path = self.config.get('models', {}).get('forecasting_model_path')

            if not model_path or not os.path.exists(model_path):            if not model_path or not os.path.exists(model_path):

                raise FileNotFoundError(f"Forecasting model not found: {model_path}")                raise FileNotFoundError(f"Forecasting model not found: {model_path}")

                        

            self.logger.info("📦 Loading trained forecasting model...")            self.logger.info("📦 Loading trained forecasting model...")

            initial_model = get_initial_model(model_path)            initial_model = get_initial_model(model_path)

                        

            # Initialize Dash plotter            # Initialize Dash plotter

            plotter = DashRealTimePlotter()            plotter = DashRealTimePlotter()

            self.logger.info("🎯 Starting Dash server...")            self.logger.info("🎯 Starting Dash server...")

            plotter.start_server()            plotter.start_server()

                        

            print("🌐 Open http://localhost:8050 in your browser to view real-time plots")            print("🌐 Open http://localhost:8050 in your browser to view real-time plots")

            time.sleep(3)  # Allow server to initialize            time.sleep(3)  # Allow server to initialize

                        

            # Main monitoring loop            # Main monitoring loop

            while True:            while True:

                try:                try:

                    self.logger.info("🔄 Fetching fresh data from Zabbix...")                    self.logger.info("🔄 Fetching fresh data from Zabbix...")

                                        

                    # Get online data from Zabbix (replaces your df_online)                    # Get online data from Zabbix (replaces your df_online)

                    df_online, scalers, context_length, variables = self.get_online_data()                    df_online, scalers, context_length, variables = self.get_online_data()

                                        

                    if df_online.empty:                    if df_online.empty:

                        self.logger.warning("⚠️  No data received, waiting for next cycle...")                        self.logger.warning("⚠️  No data received, waiting for next cycle...")

                        time.sleep(self.update_interval)                        time.sleep(self.update_interval)

                        continue                        continue

                                        

                    # Prepare data structures for forecasting                    # Prepare data structures for forecasting

                    df_removed_nans_forecasting = df_online.copy()                    df_removed_nans_forecasting = df_online.copy()

                    df_removed_nans_classification = df_online.copy()                    df_removed_nans_classification = df_online.copy()

                    prediction_horizon = self.config.get('monitoring', {}).get('prediction_horizon', 6)                    prediction_horizon = self.config.get('monitoring', {}).get('prediction_horizon', 6)

                                        

                    self.logger.info(f"🎯 Running forecasting with {len(df_online)} data points...")                    self.logger.info(f"🎯 Running forecasting with {len(df_online)} data points...")

                                        

                    # Run your existing multi-step forecasting                    # Run your existing multi-step forecasting

                    predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df = \                    predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df = \

                        multistep_rolling_buffer_learning_prediction_with_dash(                        multistep_rolling_buffer_learning_prediction_with_dash(

                            initial_model=initial_model,                            initial_model=initial_model,

                            df_online=df_online,                            df_online=df_online,

                            scalers=scalers,                            scalers=scalers,

                            context_length=context_length,                            context_length=context_length,

                            df_removed_nans_forecasting=df_removed_nans_forecasting,                            df_removed_nans_forecasting=df_removed_nans_forecasting,

                            df_removed_nans_classification=df_removed_nans_classification,                            df_removed_nans_classification=df_removed_nans_classification,

                            dash_plotter=plotter,                            dash_plotter=plotter,

                            variables=variables,                            variables=variables,

                            prediction_horizon=prediction_horizon                            prediction_horizon=prediction_horizon

                        )                        )

                                        

                    self.logger.info("✅ Forecasting cycle completed successfully")                    self.logger.info("✅ Forecasting cycle completed successfully")

                                        

                    # Wait for next cycle                    # Wait for next cycle

                    self.logger.info(f"⏱️  Waiting {self.update_interval} seconds for next cycle...")                    self.logger.info(f"⏱️  Waiting {self.update_interval} seconds for next cycle...")

                    time.sleep(self.update_interval)                    time.sleep(self.update_interval)

                                        

                except KeyboardInterrupt:                except KeyboardInterrupt:

                    self.logger.info("🛑 Keyboard interrupt received, shutting down...")                    self.logger.info("🛑 Keyboard interrupt received, shutting down...")

                    break                    break

                except Exception as e:                except Exception as e:

                    self.logger.error(f"❌ Error in forecasting loop: {e}")                    self.logger.error(f"❌ Error in forecasting loop: {e}")

                    self.logger.info(f"⏱️  Waiting {self.update_interval} seconds before retry...")                    self.logger.info(f"⏱️  Waiting {self.update_interval} seconds before retry...")

                    time.sleep(self.update_interval)                    time.sleep(self.update_interval)

                        

        except Exception as e:        except Exception as e:

            self.logger.error(f"❌ Fatal error in monitoring loop: {e}")            self.logger.error(f"❌ Fatal error in monitoring loop: {e}")

            raise            raise



def main():def main():

    """Main entry point"""    """Main entry point"""

    import argparse    import argparse

        

    parser = argparse.ArgumentParser(description='Zabbix Data Connector for Industrial Anomaly Detection')    parser = argparse.ArgumentParser(description='Zabbix Data Connector for Industrial Anomaly Detection')

    parser.add_argument('--config', '-c', default='config.json',    parser.add_argument('--config', '-c', default='config.json',

                       help='Configuration file path')                       help='Configuration file path')

    parser.add_argument('--test', '-t', action='store_true',    parser.add_argument('--test', '-t', action='store_true',

                       help='Test connection and data fetch only')                       help='Test connection and data fetch only')

        

    args = parser.parse_args()    args = parser.parse_args()

        

    try:    try:

        # Initialize connector        # Initialize connector

        connector = ZabbixDataConnector(args.config)        connector = ZabbixDataConnector(args.config)

                

        if args.test:        if args.test:

            # Test mode - just fetch data and display info            # Test mode - just fetch data and display info

            print("🧪 Running in test mode...")            print("🧪 Running in test mode...")

                        

            df_online, scalers, context_length, variables = connector.get_online_data()            df_online, scalers, context_length, variables = connector.get_online_data()

                        

            print(f"✅ Test successful!")            print(f"✅ Test successful!")

            print(f"   Data shape: {df_online.shape}")            print(f"   Data shape: {df_online.shape}")

            print(f"   Variables: {variables}")            print(f"   Variables: {variables}")

            print(f"   Context length: {context_length}")            print(f"   Context length: {context_length}")

            print(f"   Time range: {df_online.index.min()} to {df_online.index.max()}")            print(f"   Time range: {df_online.index.min()} to {df_online.index.max()}")

                        

        else:        else:

            # Production mode - run continuous monitoring            # Production mode - run continuous monitoring

            print("🏭 Starting production monitoring...")            print("🏭 Starting production monitoring...")

            connector.run_monitoring_loop()            connector.run_monitoring_loop()

                        

    except KeyboardInterrupt:    except KeyboardInterrupt:

        print("\n🛑 Shutdown requested by user")        print("\n🛑 Shutdown requested by user")

    except Exception as e:    except Exception as e:

        print(f"❌ Fatal error: {e}")        print(f"❌ Fatal error: {e}")

        sys.exit(1)        sys.exit(1)



if __name__ == "__main__":    """

    main()
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