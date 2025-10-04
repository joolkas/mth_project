#!/usr/bin/env python3
"""
🏭 Simplified Zabbix Data Collector for Industrial Anomaly Detection

Real-time data collection from Zabbix systems based on search criteria.
Designed to work with your existing LSTM forecasting models.

Author: Industrial Network Analysis System
Version: 1.0
"""

import sys
import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pyzabbix import ZabbixAPI
import urllib3

# Disable SSL warnings for internal networks
urllib3.disable_warnings()


class ZabbixDataCollector:
    """Simplified Zabbix data collector with search-based item discovery"""
    
    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        self.zabbix_api = None
        
        # Data collection parameters
        self.search_criteria = self.config['data_collection']['search_criteria']
        self.host_groups = self.config['data_collection']['host_groups']
        self.update_interval = self.config['data_collection']['update_interval']
        self.history_hours = self.config['data_collection']['history_hours']
        
        # Cache for discovered items to improve performance
        self._item_cache = {}
        self._host_cache = {}
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Config error: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """Setup logging"""
        log_level = getattr(logging, self.config['monitoring']['log_level'], logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('zabbix_collector.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """Connect to Zabbix API"""
        try:
            zabbix_config = self.config['zabbix']
            self.logger.info(f"🔗 Connecting to Zabbix: {zabbix_config['url']}")
            
            self.zabbix_api = ZabbixAPI(zabbix_config['url'])
            self.zabbix_api.session.verify = False
            self.zabbix_api.login(zabbix_config['user'], zabbix_config['password'])
            
            # Test connection
            version = self.zabbix_api.apiinfo.version()
            self.logger.info(f"✅ Connected to Zabbix {version}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Zabbix connection failed: {e}")
            return False
    
    def discover_hosts(self) -> List[Dict]:
        """Discover hosts from configured host groups"""
        if self._host_cache:
            return self._host_cache
        
        try:
            all_hosts = []
            
            for group_name in self.host_groups:
                try:
                    # Find the host group
                    groups = self.zabbix_api.hostgroup.get(filter={"name": group_name})
                    if not groups:
                        self.logger.warning(f"⚠️ Host group '{group_name}' not found")
                        continue
                    
                    # Get hosts from this group
                    hosts = self.zabbix_api.host.get(
                        groupids=[groups[0]['groupid']],
                        output=['hostid', 'host', 'name', 'status'],
                        filter={'status': 0}  # Only enabled hosts
                    )
                    
                    all_hosts.extend(hosts)
                    self.logger.info(f"📡 Found {len(hosts)} hosts in group '{group_name}'")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error accessing group '{group_name}': {e}")
                    continue
            
            # Remove duplicates and cache results
            unique_hosts = []
            seen_hostids = set()
            for host in all_hosts:
                if host['hostid'] not in seen_hostids:
                    unique_hosts.append(host)
                    seen_hostids.add(host['hostid'])
            
            self._host_cache = unique_hosts
            self.logger.info(f"🏭 Discovered {len(unique_hosts)} unique industrial hosts")
            return unique_hosts
            
        except Exception as e:
            self.logger.error(f"❌ Host discovery failed: {e}")
            return []
    
    def discover_items(self, hosts: List[Dict]) -> List[Dict]:
        """Discover monitored items based on search criteria"""
        if self._item_cache:
            return self._item_cache
        
        try:
            all_items = []
            host_ids = [host['hostid'] for host in hosts]
            
            # Get all monitored numeric items from these hosts
            items = self.zabbix_api.item.get(
                hostids=host_ids,
                output=['itemid', 'name', 'key_', 'hostid', 'value_type'],
                monitored=True,
                filter={'value_type': [0, 3]}  # Numeric float and integer values
            )
            
            # Create host lookup for performance
            host_lookup = {host['hostid']: host for host in hosts}
            
            # Filter items based on search criteria
            filtered_items = []
            for item in items:
                item_name_lower = item['name'].lower()
                item_key_lower = item['key_'].lower()
                
                # Check if any search criteria matches
                for criteria in self.search_criteria:
                    if (criteria.lower() in item_name_lower or 
                        criteria.lower() in item_key_lower):
                        
                        # Add host information
                        if item['hostid'] in host_lookup:
                            item['host_name'] = host_lookup[item['hostid']]['host']
                            item['display_name'] = f"{item['host_name']}_{item['name']}"
                            filtered_items.append(item)
                        break
            
            self._item_cache = filtered_items
            
            # Log discovery results
            self.logger.info(f"🔍 Item discovery results:")
            self.logger.info(f"   Total items found: {len(items)}")
            self.logger.info(f"   Items matching criteria: {len(filtered_items)}")
            
            criteria_counts = {}
            for item in filtered_items:
                for criteria in self.search_criteria:
                    if (criteria.lower() in item['name'].lower() or 
                        criteria.lower() in item['key_'].lower()):
                        criteria_counts[criteria] = criteria_counts.get(criteria, 0) + 1
            
            for criteria, count in criteria_counts.items():
                self.logger.info(f"   '{criteria}': {count} items")
            
            return filtered_items
            
        except Exception as e:
            self.logger.error(f"❌ Item discovery failed: {e}")
            return []
    
    def collect_history_data(self, items: List[Dict], hours_back: int = None) -> Optional[pd.DataFrame]:
        """Collect historical data for model training"""
        if hours_back is None:
            hours_back = self.history_hours
        
        try:
            time_to = int(time.time())
            time_from = time_to - (hours_back * 3600)
            
            self.logger.info(f"📊 Collecting {hours_back} hours of historical data...")
            self.logger.info(f"   Time range: {datetime.fromtimestamp(time_from)} to {datetime.fromtimestamp(time_to)}")
            
            all_data = []
            item_ids = [item['itemid'] for item in items]
            
            # Split into batches to avoid API limits
            batch_size = 50
            for i in range(0, len(item_ids), batch_size):
                batch_ids = item_ids[i:i + batch_size]
                
                try:
                    history = self.zabbix_api.history.get(
                        itemids=batch_ids,
                        time_from=time_from,
                        time_till=time_to,
                        output='extend',
                        sortfield='clock'
                    )
                    
                    # Process batch results
                    item_lookup = {item['itemid']: item for item in items}
                    
                    for record in history:
                        if record['itemid'] in item_lookup:
                            try:
                                all_data.append({
                                    'timestamp': pd.to_datetime(int(record['clock']), unit='s'),
                                    'variable': item_lookup[record['itemid']]['display_name'],
                                    'value': float(record['value'])
                                })
                            except (ValueError, TypeError):
                                continue  # Skip invalid values
                    
                    self.logger.info(f"   Processed batch {i//batch_size + 1}/{(len(item_ids)-1)//batch_size + 1}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error in batch {i//batch_size + 1}: {e}")
                    continue
            
            if not all_data:
                self.logger.warning("⚠️ No historical data collected")
                return None
            
            # Create time series DataFrame
            df = pd.DataFrame(all_data)
            df_pivot = df.pivot_table(
                index='timestamp',
                columns='variable',
                values='value',
                aggfunc='mean'
            )
            
            # Data cleaning and preparation
            df_pivot = df_pivot.sort_index()
            
            # Handle missing values
            df_pivot = df_pivot.fillna(method='ffill', limit=5)  # Forward fill up to 5 minutes
            df_pivot = df_pivot.fillna(method='bfill', limit=5)  # Backward fill up to 5 minutes
            df_pivot = df_pivot.fillna(0)  # Fill remaining with 0
            
            # Remove columns with insufficient data (>50% missing)
            missing_threshold = 0.5
            columns_to_keep = []
            for col in df_pivot.columns:
                missing_ratio = df_pivot[col].isna().sum() / len(df_pivot)
                if missing_ratio <= missing_threshold:
                    columns_to_keep.append(col)
            
            df_pivot = df_pivot[columns_to_keep]
            
            # Resample to regular 1-minute intervals
            df_resampled = df_pivot.resample('1T').mean()
            df_resampled = df_resampled.fillna(method='ffill')
            
            self.logger.info(f"✅ Historical data collected:")
            self.logger.info(f"   Shape: {df_resampled.shape}")
            self.logger.info(f"   Time range: {df_resampled.index[0]} to {df_resampled.index[-1]}")
            self.logger.info(f"   Variables: {list(df_resampled.columns)[:5]}...")
            
            return df_resampled
            
        except Exception as e:
            self.logger.error(f"❌ Historical data collection failed: {e}")
            return None
    
    def collect_recent_data(self, items: List[Dict], hours_back: int = 2) -> Optional[pd.DataFrame]:
        """Collect recent data for real-time monitoring"""
        return self.collect_history_data(items, hours_back)
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save collected data to file"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            filepath = os.path.join('data', filename)
            df.to_csv(filepath)
            
            self.logger.info(f"💾 Data saved to: {filepath}")
            self.logger.info(f"   Shape: {df.shape}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save data: {e}")
    
    def test_connection(self) -> bool:
        """Test Zabbix connection and data availability"""
        self.logger.info("🧪 Testing Zabbix connection and data availability...")
        
        if not self.connect():
            return False
        
        # Test host discovery
        hosts = self.discover_hosts()
        if not hosts:
            self.logger.error("❌ No hosts found")
            return False
        
        # Test item discovery
        items = self.discover_items(hosts)
        if not items:
            self.logger.error("❌ No items found matching search criteria")
            return False
        
        # Test data collection (small sample)
        sample_data = self.collect_recent_data(items, hours_back=1)
        if sample_data is None or sample_data.empty:
            self.logger.error("❌ No data could be collected")
            return False
        
        self.logger.info("✅ All connection tests passed!")
        self.logger.info(f"   Found {len(hosts)} hosts")
        self.logger.info(f"   Found {len(items)} matching items")
        self.logger.info(f"   Sample data shape: {sample_data.shape}")
        
        return True


def main():
    """Main entry point for data collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Zabbix Data Collector')
    parser.add_argument('--config', '-c', default='config.json', help='Configuration file')
    parser.add_argument('--test', '-t', action='store_true', help='Test connection only')
    parser.add_argument('--collect-history', action='store_true', help='Collect historical data for training')
    parser.add_argument('--hours', type=int, default=48, help='Hours of history to collect')
    
    args = parser.parse_args()
    
    collector = ZabbixDataCollector(args.config)
    
    if args.test:
        success = collector.test_connection()
        sys.exit(0 if success else 1)
    
    # Connect to Zabbix
    if not collector.connect():
        sys.exit(1)
    
    # Discover hosts and items
    hosts = collector.discover_hosts()
    if not hosts:
        print("❌ No hosts found")
        sys.exit(1)
    
    items = collector.discover_items(hosts)
    if not items:
        print("❌ No items found")
        sys.exit(1)
    
    if args.collect_history:
        # Collect historical data for model training
        print(f"📊 Collecting {args.hours} hours of historical data...")
        df = collector.collect_history_data(items, args.hours)
        
        if df is not None:
            # Save the data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"historical_data_{timestamp}.csv"
            collector.save_data(df, filename)
            print(f"✅ Historical data saved: {filename}")
        else:
            print("❌ Failed to collect historical data")
            sys.exit(1)
    else:
        # Collect recent data for monitoring
        print("📡 Collecting recent data...")
        df = collector.collect_recent_data(items)
        
        if df is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recent_data_{timestamp}.csv"
            collector.save_data(df, filename)
            print(f"✅ Recent data saved: {filename}")
        else:
            print("❌ Failed to collect recent data")
            sys.exit(1)


if __name__ == "__main__":
    main()