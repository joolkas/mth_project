#!/usr/bin/env python3
"""
Simple Zabbix Data Collector for Industrial Network Anomaly Detection
Collects data from Zabbix API for model training and real-time monitoring
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pyzabbix import ZabbixAPI
import urllib3

# Disable SSL warnings for internal networks
urllib3.disable_warnings()


class ZabbixDataCollector:
    """Simple Zabbix data collector with minimal dependencies"""
    
    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        self.zabbix_api = None
        
        # Configuration
        self.zabbix_config = self.config['zabbix']
        self.search_criteria = self.config['data_collection']['search_criteria']
        self.host_groups = self.config['data_collection']['host_groups']
        self.history_hours = self.config['data_collection']['history_hours']
        
        # Timezone configuration
        self.server_utc_offset = self.config.get('timezone', {}).get('server_utc_offset', 0)
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def connect(self) -> bool:
        """Connect to Zabbix API"""
        try:
            print(f"Connecting to Zabbix: {self.zabbix_config['url']}")
            self.zabbix_api = ZabbixAPI(self.zabbix_config['url'])
            self.zabbix_api.session.verify = False
            self.zabbix_api.login(self.zabbix_config['user'], self.zabbix_config['password'])
            
            version = self.zabbix_api.apiinfo.version()
            print(f"Connected to Zabbix {version}")
            return True
            
        except Exception as e:
            print(f"Zabbix connection failed: {e}")
            return False
    
    def discover_items(self) -> List[Dict]:
        """Discover monitored items based on search criteria"""
        try:
            # Get hosts from configured groups
            all_hosts = []
            for group_name in self.host_groups:
                groups = self.zabbix_api.hostgroup.get(filter={"name": group_name})
                if groups:
                    hosts = self.zabbix_api.host.get(
                        groupids=[groups[0]['groupid']],
                        output=['hostid', 'host', 'name'],
                        filter={'status': 0}  # Only enabled hosts
                    )
                    all_hosts.extend(hosts)
            
            if not all_hosts:
                print("No hosts found")
                return []
            
            # Get monitored numeric items with extended info for debugging
            host_ids = [host['hostid'] for host in all_hosts]
            items = self.zabbix_api.item.get(
                hostids=host_ids,
                output=['itemid', 'name', 'key_', 'hostid', 'value_type', 'units', 'preprocessing'],
                monitored=True,
                filter={'value_type': [0, 3]}  # Numeric values only
            )
            
            # Filter items based on search criteria
            host_lookup = {host['hostid']: host for host in all_hosts}
            filtered_items = []
            
            for item in items:
                item_name_lower = item['name'].lower()
                item_key_lower = item['key_'].lower()
                
                print(f"DEBUG FOR DATA COLLECTION: {item_name_lower}, {item_key_lower}")
                # Check if any search criteria matches
                for criteria in self.search_criteria:
                    print(f"DEBUG DATA COLLECTION: {criteria}")
                    if (criteria.lower() in item_name_lower or 
                        criteria.lower() in item_key_lower):
                        if item['hostid'] in host_lookup:
                            item['host_name'] = host_lookup[item['hostid']]['host']
                            item['display_name'] = f"{item['host_name']}_{item['name']}"
                            filtered_items.append(item)
                        break
            
            # print(f"Found {len(filtered_items)} matching items")
            return filtered_items
            
        except Exception as e:
            print(f"Item discovery failed: {e}")
            return []
    
    def collect_historical_data(self, items: List[Dict], hours_back: int = None) -> Optional[pd.DataFrame]:
        """Collect historical data with comprehensive debugging"""
        if hours_back is None:
            hours_back = self.history_hours
            
        try:
            time_to = int(time.time())
            time_from = int(time_to - (hours_back * 3600))  # Ensure integer timestamp

            print(f"🔍 DEBUGGING DATA COLLECTION:")
            print(f"   Collecting {hours_back} hours of historical data...")
            print(f"   Time range: {datetime.fromtimestamp(time_from)} to {datetime.fromtimestamp(time_to)}")
            print(f"   Items to collect: {len(items)}")

            all_data = []
            item_ids = [item['itemid'] for item in items]
            
            # Show what items we're requesting with detailed info
            print(f"   Item details:")
            for i, item in enumerate(items[:5]):  # Show first 5 items
                units = item.get('units', 'N/A')
                value_type = item.get('value_type', 'N/A')
                print(f"     {i+1}. {item['display_name']} (ID: {item['itemid']})")
                print(f"        Key: {item['key_']}, Type: {value_type}, Units: {units}")
                # Check if it's a rate/delta item that might cause zeros
                if 'bits' in item['name'].lower():
                    print(f"        ⚠️  Network rate item - may show zeros during low traffic")
            if len(items) > 5:
                print(f"     ... and {len(items)-5} more items")
            
            # Get history data
            print(f"   🌐 Requesting history from Zabbix API...")
            history = self.zabbix_api.history.get(
                itemids=item_ids,
                time_from=time_from,
                time_till=time_to,
                output='extend',
                sortfield='clock'
            )
            
            print(f"   📊 Received {len(history)} raw data records from Zabbix")
            
            # Process data with detailed debugging
            item_lookup = {item['itemid']: item for item in items}
            processed_count = 0
            error_count = 0
            value_examples = {}
            
            for record in history:
                if record['itemid'] in item_lookup:
                    try:
                        # Convert Zabbix timestamp (UTC) to server local time
                        # Apply server UTC offset from config
                        timestamp = pd.to_datetime(int(record['clock']) + (self.server_utc_offset * 3600), unit='s')
                        value = float(record['value'])
                        variable_name = item_lookup[record['itemid']]['display_name']
                        
                        all_data.append({
                            'timestamp': timestamp,
                            'variable': variable_name,
                            'value': value
                        })
                        
                        processed_count += 1
                        
                        # Collect value examples for debugging
                        if variable_name not in value_examples:
                            value_examples[variable_name] = []
                        if len(value_examples[variable_name]) < 5:
                            value_examples[variable_name].append((timestamp, value))
                            
                    except (ValueError, TypeError) as e:
                        error_count += 1
                        if error_count <= 5:  # Show first 5 errors
                            print(f"     ⚠️  Error processing record: {record}, Error: {e}")
                        continue
            
            print(f"   ✅ Processed {processed_count} records, {error_count} errors")
            
            # Show sample values for debugging - ENHANCED
            print(f"   🔍 Sample values collected:")
            for var_name, examples in list(value_examples.items())[:3]:  # Show first 3 variables
                print(f"     {var_name}:")
                for ts, val in examples:
                    print(f"       {ts}: {val}")
                    
            # Additional debugging for zero values
            print(f"   🔍 ZERO VALUE ANALYSIS:")
            for var_name, examples in value_examples.items():
                zero_count = sum(1 for _, val in examples if val == 0.0)
                if zero_count > 0:
                    print(f"     ⚠️  {var_name}: {zero_count}/{len(examples)} samples are zero!")
                    
            # Show raw value distribution for network interfaces
            print(f"   🔍 RAW VALUE DISTRIBUTION (before any processing):")
            network_vars = [v for v in value_examples.keys() if 'bits' in v.lower() or 'interface' in v.lower()]
            for var_name in network_vars[:2]:  # Show first 2 network variables
                if var_name in value_examples:
                    values = [val for _, val in value_examples[var_name]]
                    print(f"     {var_name}: min={min(values):.1f}, max={max(values):.1f}, mean={sum(values)/len(values):.1f}")
            
            if not all_data:
                print("   ❌ No valid data collected after processing")
                return None
            
            if not all_data:
                print("   ❌ No valid data collected after processing")
                return None
            
            # Create time series DataFrame
            print(f"   📋 Creating DataFrame from {len(all_data)} records...")
            df = pd.DataFrame(all_data)
            print(f"   📋 Raw DataFrame shape: {df.shape}")
            
            # Show data distribution
            print(f"   📊 Data distribution by variable:")
            for var in df['variable'].unique()[:5]:  # Show first 5 variables
                count = len(df[df['variable'] == var])
                print(f"     {var}: {count} records")
            
            df_pivot = df.pivot_table(
                index='timestamp',
                columns='variable', 
                values='value',
                aggfunc='last'  # Use 'last' to avoid averaging issues
            )
            
            print(f"   📊 Pivot table shape: {df_pivot.shape}")
            print(f"   📅 Time range: {df_pivot.index[0]} to {df_pivot.index[-1]}")
            
            # Show sample pivoted data before cleaning
            print(f"   🔍 Sample pivoted data (before cleaning):")
            if len(df_pivot) > 0:
                sample_idx = min(5, len(df_pivot))
                for col in df_pivot.columns[:3]:  # Show first 3 columns
                    print(f"     {col}: {df_pivot[col].head(sample_idx).tolist()}")
            
            # Clean and resample data with improved handling
            df_pivot = df_pivot.sort_index()
            
            # Remove any infinite values that might cause issues
            inf_count = np.isinf(df_pivot.values).sum()
            print(f"   🧹 Removing {inf_count} infinite values")
            df_pivot = df_pivot.replace([np.inf, -np.inf], np.nan)
            
            # Fill missing values more conservatively
            nan_before = df_pivot.isna().sum().sum()
            df_pivot = df_pivot.fillna(method='ffill', limit=2)  # Reduced limit
            df_pivot = df_pivot.fillna(method='bfill', limit=2)  # Reduced limit
            df_pivot = df_pivot.fillna(0)
            nan_after = df_pivot.isna().sum().sum()
            print(f"   🧹 Filled {nan_before - nan_after} missing values")
            
            # Smart resampling with "last known good value" mechanism
            print(f"   ⏰ Smart resampling with last-known-good-value logic...")
            
            # Show original data frequency
            if len(df_pivot) > 1:
                time_diff = df_pivot.index[1] - df_pivot.index[0]
                print(f"     Original data frequency: ~{time_diff}")
            
            # Create 1-minute index for the full time range
            full_time_range = pd.date_range(start=df_pivot.index[0], end=df_pivot.index[-1], freq='1T')
            df_resampled = pd.DataFrame(index=full_time_range, columns=df_pivot.columns)
            
            # For each column, implement last-known-good-value logic
            print(f"   🔄 Applying last-known-good-value logic for each variable...")
            for col in df_pivot.columns:
                original_data = df_pivot[col].dropna()  # Remove NaN values
                print(f"     Processing {col}: {len(original_data)} real data points")
                
                if len(original_data) == 0:
                    continue
                    
                # Fill with last-known-good values
                for timestamp in df_resampled.index:
                    # Find the most recent real value before or at this timestamp
                    available_data = original_data[original_data.index <= timestamp]
                    
                    if len(available_data) > 0:
                        # Use the last known good value
                        last_good_value = available_data.iloc[-1]
                        df_resampled.loc[timestamp, col] = last_good_value
                    else:
                        # No data available yet, use first available value
                        df_resampled.loc[timestamp, col] = original_data.iloc[0]
            
            # Verify and clean up any remaining issues
            print(f"   🔧 Final cleanup and validation...")
            
            # Check for any remaining NaN values
            for col in df_resampled.columns:
                nan_count = df_resampled[col].isna().sum()
                if nan_count > 0:
                    print(f"     ⚠️  {col}: {nan_count} still missing - using column median")
                    median_value = df_resampled[col].median()
                    if pd.isna(median_value):
                        median_value = 0  # Fallback if all values are NaN
                    df_resampled[col] = df_resampled[col].fillna(median_value)
            
            # Ensure all values are finite
            df_resampled = df_resampled.replace([np.inf, -np.inf], np.nan)
            df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Final analysis with last-known-good validation
            print(f"   🌐 Final data quality check with last-known-good values:")
            network_columns = [col for col in df_resampled.columns if 'bits' in col.lower() or 'interface' in col.lower()]
            
            for col in network_columns[:2]:  # Show first 2 network columns
                # Check for artificial zeros vs real zeros
                original_data = df_pivot[col].dropna()
                if len(original_data) > 0:
                    print(f"     {col}:")
                    print(f"       Original data points: {len(original_data)}")
                    print(f"       Resampled to: {len(df_resampled[col])} points")
                    
                    # Show value stability - should have fewer unique values now
                    unique_resampled = df_resampled[col].nunique()
                    unique_original = original_data.nunique()
                    print(f"       Value stability: {unique_original} unique → {unique_resampled} unique")
                    
                    # Show recent values to verify last-known-good logic
                    print(f"       Last 3 values: {df_resampled[col].tail(3).tolist()}")
                    
            print(f"   ✅ Last-known-good-value resampling completed")
            
            # Final cleanup
            df_resampled = df_resampled.fillna(0)
            df_resampled = df_resampled.replace([np.inf, -np.inf], 0)
            
            print(f"   ✅ Final resampled data shape: {df_resampled.shape}")
            
            # Show sample final data
            print(f"   🔍 Sample final data (last 3 rows):")
            if len(df_resampled) > 0:
                for col in df_resampled.columns[:3]:  # Show first 3 columns
                    last_values = df_resampled[col].tail(3).tolist()
                    print(f"     {col}: {last_values}")

            print(f"✅ Historical data collection completed: {df_resampled.shape}")
            return df_resampled
            
        except Exception as e:
            print(f"Historical data collection failed: {e}")
            return None
    def collect_recent_data(self, items: List[Dict], hours_back: int = 2) -> Optional[pd.DataFrame]:
        """Collect recent data for real-time monitoring"""
        return self.collect_historical_data(items, hours_back)
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save data to CSV file"""
        try:
            import os
            os.makedirs('data', exist_ok=True)
            filepath = f"data/{filename}"
            df.to_csv(filepath)
            print(f"Data saved to: {filepath}")
        except Exception as e:
            print(f"Failed to save data: {e}")


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Zabbix Data Collector')
    parser.add_argument('--test', action='store_true', help='Test connection only')
    parser.add_argument('--collect', action='store_true', help='Collect historical data')
    parser.add_argument('--hours', type=int, default=48, help='Hours of history to collect')
    
    args = parser.parse_args()
    
    collector = ZabbixDataCollector()
    
    if not collector.connect():
        return 1
    
    items = collector.discover_items()
    if not items:
        print("No items found")
        return 1
    
    if args.test:
        print(f"✅ Connection test successful - found {len(items)} items")
        return 0
    
    if args.collect:
        df = collector.collect_historical_data(items, args.hours)
        if df is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.csv"
            collector.save_data(df, filename)
            print(f"✅ Data collected and saved: {filename}")
        else:
            print("❌ Failed to collect data")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())