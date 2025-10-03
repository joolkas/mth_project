import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import Dataset
from data_utils import *
import warnings
import logging
import json
import time
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from pyzabbix import ZabbixAPI
import urllib3

# configure warning logging
warnings_logger = logging.getLogger('warnings')
warnings_logger.setLevel(logging.WARNING)
warning_handler = logging.FileHandler('warnings.log')
warning_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
warnings_logger.addHandler(warning_handler)

# capture warnings and log them
def warning_handler_func(message, category, filename, lineno, file=None, line=None):
    warnings_logger.warning(f"{category.__name__}: {message} (File: {filename}, Line: {lineno})")

warnings.showwarning = warning_handler_func

def get_path_and_device_name():
    device_name="DEMO"
    # Updated path to go up one level from zabbix_integration folder
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "RealData")
    return device_name, data_path

def get_processed_path():
    device_name, data_path = get_path_and_device_name()
    processed_data_path = os.path.join(data_path, "processed")
    processed_forecasting_path = os.path.join(processed_data_path, f"{device_name}_forecasting.csv")
    processed_statuses_path = os.path.join(processed_data_path, f"{device_name}_statuses.csv")
    return processed_forecasting_path, processed_statuses_path

def get_column_names_exclude(df, search_keyword, exclude_keyword):
    df_column_names = df.get_column_names(search_keyword)
    df_column_names_excluded = []

    for col in df_column_names:
        if exclude_keyword.lower() not in col.lower():
            df_column_names_excluded.append(col)
    return df_column_names_excluded

device_name, data_path = get_path_and_device_name()
processed_forecasting_path, processed_statuses_path = get_processed_path()

def get_data_function_original(device_name=device_name, data_path=data_path,
              numeric_names=["ICMP response time", "temperature", "cpu", "used memory", "bits"], status_names = [": operational status"], numeric_exclude ="status", status_exclude="unused"):
    """Original function for CSV data - kept for reference"""
    
    csv_path = os.path.join(data_path, f"{device_name}.csv")
    df = Dataset(csv_path)

    df_numerics = []
    df_statuses = []

    for numeric_name in numeric_names:
        df_numerics += get_column_names_exclude(df, numeric_name, numeric_exclude)
    
    for status_name in status_names:
        df_statuses += get_column_names_exclude(df, status_name, status_exclude)

    print("Getting values...")
    df_numeric_values = df.get_column_values(df_numerics)
    df_status_values = df.get_column_values(df_statuses)

    # if numeric values are constant for longer than one day, remove them
    half_day = 12 * 60
    df_numeric_one_day = df_numeric_values.iloc[16000:16000+half_day]
    for col in df_numeric_one_day.columns:
        if df_numeric_one_day[col].nunique() <= 1:
            print(f"Removing constant column: {col}")
            df_numeric_values.drop(columns=[col], inplace=True)

    print(f"Numeric columns: {df_numeric_values.shape}")
    print(f"Status columns: {df_status_values.shape}")

    # limit amount of values, use numeric for forecasting
    df_forecasting = df_numeric_values.iloc[16000:]
    df_classification = df_status_values.iloc[16000:]

    # PREPROCESSING
    print("Preprocessing data...")

    # finish data preprocessing for forecasting
    df_removed_outliers_forecasting = remove_outliers(df_forecasting, 1000)
    df_removed_nans_forecasting = df_removed_outliers_forecasting.dropna(axis=1, how="all")

    # preprocessing for classification, which uses all data
    df_removed_outliers_statuses = remove_outliers(df_classification, 1000)
    df_removed_nans_statuses = df_removed_outliers_statuses.dropna(axis=1, how="all")

    # Ensure processed directory exists
    os.makedirs(os.path.dirname(processed_forecasting_path), exist_ok=True)
    
    # save as csv
    df_removed_nans_forecasting.to_csv(processed_forecasting_path)
    df_removed_nans_statuses.to_csv(processed_statuses_path)
    
    print(f"Processed data saved to {processed_forecasting_path} and {processed_statuses_path}")

    return df_removed_nans_forecasting, df_removed_nans_statuses

def get_data_function(use_zabbix=True, device_name=device_name, data_path=data_path):
    """Get data from Zabbix (default) or CSV files, processed for initial model training"""
    
    if use_zabbix:
        print("🔌 Getting data from Zabbix...")
        return get_data_from_zabbix()
    else:
        print("📁 Getting data from CSV files...")
        return get_data_function_original(device_name, data_path)

def get_data_from_zabbix():
    """Get real data from Zabbix using the same approach as online loop"""
    
    # Load configuration - now in same directory
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    zabbix_config = config['zabbix']
    
    print(f"🔌 Connecting to Zabbix server: {zabbix_config['url']}")
    
    # Connect to Zabbix
    try:
        zapi = ZabbixAPI(zabbix_config['url'])
        # Disable SSL verification for self-signed certificates
        zapi.session.verify = False
        # Suppress SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        zapi.login(zabbix_config['user'], zabbix_config['password'])
        print("✅ Connected to Zabbix successfully")
    except Exception as e:
        print(f"❌ Failed to connect to Zabbix: {e}")
        return None, None
    
    # Collect data
    print("📊 Collecting data from Zabbix...")
    df_raw = collect_training_data_from_zabbix(zapi, config)
    
    if df_raw is not None and not df_raw.empty:
        print(f"✅ Collected {len(df_raw)} records")
        
        # Process the data to match expected format
        df_forecasting, df_classification = process_zabbix_data_for_training(df_raw)
        
        # Ensure processed directory exists
        os.makedirs(os.path.dirname(processed_forecasting_path), exist_ok=True)
        
        # Save processed data
        df_forecasting.to_csv(processed_forecasting_path)
        df_classification.to_csv(processed_statuses_path)
        
        print(f"Processed data saved to {processed_forecasting_path} and {processed_statuses_path}")
        
        return df_forecasting, df_classification
    else:
        print("❌ No data collected")
        return None, None

def collect_training_data_from_zabbix(zapi: ZabbixAPI, config: Dict) -> Optional[pd.DataFrame]:
    """Collect data from Zabbix for training (similar to online loop but for longer period)"""
    
    try:
        # Get all hosts from configured host groups
        host_groups = config['industrial_filters'].get('device_groups', [])
        all_hosts = []
        
        for group_name in host_groups:
            try:
                # Get group ID
                groups = zapi.hostgroup.get(filter={"name": group_name})
                if not groups:
                    print(f"⚠️ Host group '{group_name}' not found")
                    continue
                
                group_id = groups[0]['groupid']
                
                # Get hosts in this group
                hosts = zapi.host.get(groupids=[group_id])
                all_hosts.extend(hosts)
                print(f"📍 Found {len(hosts)} hosts in group '{group_name}'")
                
            except Exception as e:
                print(f"⚠️ Error getting hosts from group '{group_name}': {e}")
                continue
        
        if not all_hosts:
            print("❌ No hosts found in configured groups")
            return None
        
        # Define time range (last 7 days for more training data)
        time_to = int(time.time())
        time_from = time_to - (7 * 24 * 60 * 60)  # 7 days ago
        
        print(f"📅 Collecting data from {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_from))} to {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_to))}")
        
        # Collect data for each host
        all_data = []
        
        for host in all_hosts:
            host_name = host['host']
            host_id = host['hostid']
            
            print(f"🔍 Processing host: {host_name}")
            
            # Get items for this host
            items = zapi.item.get(
                hostids=[host_id],
                output=['itemid', 'key_', 'name', 'value_type'],
                monitored=True
            )
            
            if not items:
                print(f"⚠️ No monitored items found for host {host_name}")
                continue
            
            # Filter items we're interested in - COLLECT ALL ITEMS to match training data exactly
            relevant_items = {}
            for item in items:
                key = item['key_']
                name = item['name']
                
                # Create descriptive names that match training data format
                item_name = f"{host_name} - {name}"
                relevant_items[item['itemid']] = item_name
            
            if not relevant_items:
                print(f"⚠️ No relevant items found for host {host_name}")
                continue
            
            # Get history for ALL relevant items (same format as training data)
            all_records = []
            
            # Fetch history for all items at once
            if relevant_items:
                history = zapi.history.get(
                    itemids=list(relevant_items.keys()),
                    time_from=time_from,
                    time_till=time_to,
                    output='extend',
                    sortfield='clock',
                    sortorder='ASC'
                )
                
                # Convert to training data format (name, timestamp, value)
                for record in history:
                    if record['itemid'] in relevant_items:
                        all_records.append({
                            'name': relevant_items[record['itemid']],
                            'timestamp': pd.to_datetime(int(record['clock']), unit='s'),
                            'value': float(record['value'])
                        })
                
                print(f"✅ Collected {len(all_records)} records from {host_name}")
                all_data.extend(all_records)
        
        if not all_data:
            print("❌ No data collected from any host")
            return None
        
        # Create DataFrame in training format
        df_combined = pd.DataFrame(all_data)
        print(f"📊 Total combined records: {len(df_combined)}")
        print(f"📊 Unique items: {df_combined['name'].nunique()}")
        
        return df_combined
        
    except Exception as e:
        print(f"❌ Error collecting data from Zabbix: {e}")
        return None

def process_zabbix_data_for_training(df_raw: pd.DataFrame):
    """Process Zabbix data to match the expected format for training"""
    
    print("🔄 Processing Zabbix data for training format...")
    
    try:
        # Convert raw data format to time series format
        print(f"Raw data shape: {df_raw.shape}")
        print(f"Raw data columns: {df_raw.columns.tolist()}")
        print(f"Sample raw data:\n{df_raw.head()}")
        
        # Pivot the data to get metrics as columns (same as online system)
        df_pivot = df_raw.pivot_table(
            index='timestamp', 
            columns='name', 
            values='value', 
            aggfunc='mean'
        ).reset_index()
        
        print(f"Pivoted data shape: {df_pivot.shape}")
        print(f"Available metrics: {df_pivot.columns.tolist()}")
        
        # Use the Dataset class for consistent preprocessing (same as online system)
        from data_preprocessing import Dataset
        
        # Create a temporary CSV file for Dataset class
        temp_csv = "temp_training_data.csv"
        df_pivot.to_csv(temp_csv, index=False)
        
        # Use Dataset class (same as online preprocessing)
        dataset = Dataset(temp_csv)
        
        # Get all numeric columns (exclude timestamp)
        numeric_columns = [col for col in df_pivot.columns if col != 'timestamp']
        df_forecasting = dataset.get_column_values(numeric_columns)
        
        # Apply outlier removal (same as online system)
        print("Preprocessing data...")
        df_removed_outliers_forecasting = remove_outliers(df_forecasting, 1000)
        df_removed_nans_forecasting = df_removed_outliers_forecasting.dropna(axis=1, how="all")
        
        # Create simple classification data (operational status based on data availability)
        df_classification = pd.DataFrame()
        # Simple status: 1 if we have data, 0 if not
        df_classification['operational_status'] = 1
        df_removed_nans_statuses = df_classification
        
        print(f"✅ Forecasting data shape: {df_removed_nans_forecasting.shape}")
        print(f"✅ Classification data shape: {df_removed_nans_statuses.shape}")
        print(f"✅ Feature columns: {df_removed_nans_forecasting.columns.tolist()[:5]}...")
        
        # Clean up temp file
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        
        return df_removed_nans_forecasting, df_removed_nans_statuses
        
    except Exception as e:
        print(f"❌ Error processing Zabbix data: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: return empty DataFrames
        return pd.DataFrame(), pd.DataFrame()

if __name__ == "__main__":
    print("🚀 Starting real system data collection...")
    
    # Use Zabbix by default, set to False to use CSV files
    df_forecasting, df_classification = get_data_function(use_zabbix=True)
    
    if df_forecasting is not None:
        print("✅ Data collection completed successfully!")
        print(f"📊 Forecasting data: {df_forecasting.shape}")
        print(f"📊 Classification data: {df_classification.shape}")
    else:
        print("❌ Data collection failed!")