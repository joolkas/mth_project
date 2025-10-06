#!/usr/bin/env python3
"""
🔄 Simple Training Data Collection from Zabbix

Ultra-simplified version that automatically collects and processes data for model training.
"""

import sys
import os
import json
import time
import pandas as pd
from pyzabbix import ZabbixAPI
import urllib3
urllib3.disable_warnings()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data_utils import remove_outliers
    from data_preprocessing import Dataset
except ImportError as e:
    print(f"⚠️ Import error: {e}")


def collect_training_data(config_path="config.json", days=7):
    """Collect training data from Zabbix - automatically handles everything"""
    
    print(f"🔄 Collecting {days} days of training data from Zabbix...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Connect to Zabbix
    zabbix_config = config['zabbix']
    zapi = ZabbixAPI(zabbix_config['url'])
    zapi.session.verify = False
    zapi.login(zabbix_config['user'], zabbix_config['password'])
    print("✅ Connected to Zabbix")
    
    # Get hosts
    host_groups = config['industrial_filters'].get('device_groups', [])
    all_hosts = []
    
    for group_name in host_groups:
        groups = zapi.hostgroup.get(filter={"name": group_name})
        if groups:
            hosts = zapi.host.get(groupids=[groups[0]['groupid']])
            all_hosts.extend(hosts)
            print(f"📍 Found {len(hosts)} hosts in '{group_name}'")
    
    if not all_hosts:
        print("❌ No hosts found")
        return None, None
    
    # Time range
    time_to = int(time.time())
    time_from = time_to - (days * 24 * 60 * 60)
    
    print(f"📅 Collecting from {time.strftime('%Y-%m-%d', time.localtime(time_from))} to {time.strftime('%Y-%m-%d', time.localtime(time_to))}")
    
    # Collect all data
    all_data = []
    
    for host in all_hosts:
        host_name = host['host']
        print(f"🔍 Processing {host_name}")
        
        # Get all items
        items = zapi.item.get(
            hostids=[host['hostid']],
            output=['itemid', 'name'],
            monitored=True
        )
        
        if not items:
            continue
        
        # Get history
        item_ids = [item['itemid'] for item in items]
        history = zapi.history.get(
            itemids=item_ids,
            time_from=time_from,
            time_till=time_to,
            output='extend'
        )
        
        # Process data
        item_names = {item['itemid']: f"{host_name} - {item['name']}" for item in items}
        
        for record in history:
            if record['itemid'] in item_names:
                all_data.append({
                    'name': item_names[record['itemid']],
                    'timestamp': pd.to_datetime(int(record['clock']), unit='s'),
                    'value': float(record['value'])
                })
        
        print(f"✅ Collected {len([r for r in all_data if host_name in r['name']])} records from {host_name}")
    
    if not all_data:
        print("❌ No data collected")
        return None, None
    
    print(f"📊 Total records: {len(all_data)}")
    
    # Create time series DataFrame
    df = pd.DataFrame(all_data)
    df_pivot = df.pivot_table(
        index='timestamp',
        columns='name',
        values='value',
        aggfunc='mean'
    ).fillna(method='ffill').fillna(0)
    
    print(f"📊 Time series shape: {df_pivot.shape}")
    print(f"🏷️ Features: {list(df_pivot.columns)[:5]}...")
    
    # Simple preprocessing
    print("🔄 Preprocessing...")
    
    # Remove outliers
    df_clean = remove_outliers(df_pivot, 1000) if len(df_pivot) > 1000 else df_pivot
    
    # Remove empty columns
    df_final = df_clean.dropna(axis=1, how='all')
    
    print(f"✅ Final data shape: {df_final.shape}")
    
    # Save data in the EXACT format expected by initial_model.py
    
    # Create the RealData directory structure that get_processed_path() expects
    real_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "RealData")
    processed_dir = os.path.join(real_data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Use the exact file names expected by get_processed_path()
    device_name = "DEMO"  # This matches what get_processed_path() uses
    forecasting_path = os.path.join(processed_dir, f"{device_name}_forecasting.csv")
    classification_path = os.path.join(processed_dir, f"{device_name}_statuses.csv")
    
    # Save with proper datetime index (required by initial_model.py)
    df_final.to_csv(forecasting_path, index=True)  # Keep datetime index
    
    # Simple classification data (operational status)
    df_classification = pd.DataFrame({
        'operational_status': [1] * len(df_final)
    }, index=df_final.index)
    df_classification.to_csv(classification_path, index=True)  # Keep datetime index
    
    print(f"💾 Saved to {forecasting_path} and {classification_path}")
    print(f"📁 Data structure matches initial_model.py requirements")
    
    # ALSO save variables to the model directory for online system
    try:
        # Save variables list to model directory (for online monitoring)
        model_dir = "/opt/anomaly_detection/models/forecasting_model"
        if os.path.exists(model_dir):
            variables_path = os.path.join(model_dir, "variables.txt")
            with open(variables_path, 'w') as f:
                for col in df_final.columns:
                    f.write(f"{col}\n")
            print(f"🏷️ Saved {len(df_final.columns)} variables to {variables_path}")
        else:
            print("⚠️ Model directory not found - variables not saved")
    except Exception as e:
        print(f"⚠️ Could not save variables to model directory: {e}")
    
    # Save a copy in current directory for reference
    local_variables_path = "current_variables.txt"
    with open(local_variables_path, 'w') as f:
        for col in df_final.columns:
            f.write(f"{col}\n")
    print(f"📋 Reference variables saved to {local_variables_path}")
    
    return df_final, df_classification


if __name__ == "__main__":
    print("🚀 Starting simple training data collection...")
    
    # Collect data
    df_forecasting, df_classification = collect_training_data()
    
    if df_forecasting is not None:
        print("✅ Training data collection completed!")
        print(f"📊 Ready to train model with {df_forecasting.shape[1]} features")
        print("\n🎯 Next steps:")
        print("1. Use training_data/forecasting.csv to train your model")
        print("2. Copy the trained model to your deployment directory")
        print("3. Run the monitoring system")
    else:
        print("❌ Training data collection failed!")