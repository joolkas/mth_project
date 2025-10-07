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
            
            # Get monitored numeric items
            host_ids = [host['hostid'] for host in all_hosts]
            items = self.zabbix_api.item.get(
                hostids=host_ids,
                output=['itemid', 'name', 'key_', 'hostid'],
                monitored=True,
                filter={'value_type': [0, 3]}  # Numeric values only
            )
            
            # Filter items based on search criteria
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
            
            print(f"Found {len(filtered_items)} matching items")
            return filtered_items
            
        except Exception as e:
            print(f"Item discovery failed: {e}")
            return []
    
    def collect_historical_data(self, items: List[Dict], hours_back: int = None) -> Optional[pd.DataFrame]:
        """Collect historical data for training"""
        if hours_back is None:
            hours_back = self.history_hours
            
        try:
            time_to = int(time.time())
            time_from = time_to - (hours_back * 3600)

            print(f"Collecting {hours_back} hours of historical data...")

            all_data = []
            item_ids = [item['itemid'] for item in items]
            
            # Get history data
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
                        # Convert Zabbix timestamp (UTC) to server local time
                        # Apply server UTC offset from config
                        timestamp = pd.to_datetime(int(record['clock']) + (self.server_utc_offset * 3600), unit='s')
                        all_data.append({
                            'timestamp': timestamp,
                            'variable': item_lookup[record['itemid']]['display_name'],
                            'value': float(record['value'])
                        })
                    except (ValueError, TypeError):
                        continue
            
            if not all_data:
                print("No historical data collected")
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

            print(f"Historical data collected: {df_resampled.shape}")
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