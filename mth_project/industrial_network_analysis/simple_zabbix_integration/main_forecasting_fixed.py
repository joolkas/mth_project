#!/usr/bin/env python3
"""
🏭 Simplified Industrial Network Anomaly Detection - Fixed Version

This is a streamlined version of your main_forecasting.py that addresses the dashboard display issues
by creating a simple, reliable dashboard without external dependencies.

Key improvements:
1. Simplified dashboard with inline implementation
2. Better error handling and logging
3. Removed complex import dependencies
4. Fixed data update mechanism
5. More reliable threading

Author: Industrial Network Analysis System
Version: 2.0 (Simplified and Fixed)
"""

import sys
import os
import json
import time
import logging
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pyzabbix import ZabbixAPI
import urllib3

# Dashboard imports (with fallback)
try:
    import dash
    from dash import dcc, html, Input, Output
    import plotly.graph_objs as go
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("⚠️ Dashboard not available. Install with: pip install dash plotly")

# Disable warnings
urllib3.disable_warnings()

class SimpleAnomalyDetector:
    """Statistical anomaly detection using Z-score method"""
    
    def __init__(self, window_size=30, threshold=2.5):
        self.window_size = window_size
        self.threshold = threshold
        self.data_history = {}
    
    def detect_anomalies(self, data_dict: Dict[str, float]) -> Dict[str, bool]:
        """Detect anomalies in the current data"""
        anomalies = {}
        
        for variable, value in data_dict.items():
            # Initialize history if needed
            if variable not in self.data_history:
                self.data_history[variable] = deque(maxlen=self.window_size)
            
            # Add current value to history
            self.data_history[variable].append(value)
            
            # Need at least 10 points for reliable detection
            if len(self.data_history[variable]) >= 10:
                historical_data = list(self.data_history[variable])[:-1]  # Exclude current point
                mean_val = np.mean(historical_data)
                std_val = np.std(historical_data)
                
                if std_val > 0:
                    z_score = abs((value - mean_val) / std_val)
                    anomalies[variable] = z_score > self.threshold
                else:
                    anomalies[variable] = False
            else:
                anomalies[variable] = False
        
        return anomalies


class SimpleDashboard:
    """Simple, reliable dashboard implementation"""
    
    def __init__(self, port=8052):
        self.port = port
        self.app = None
        self.server_thread = None
        self.is_running = False
        
        # Thread-safe data storage
        self.data_lock = threading.Lock()
        self.timestamps = deque(maxlen=100)
        self.data_series = {}
        self.anomaly_series = {}
        
        # System stats
        self.stats = {
            'cycles': 0,
            'anomalies_detected': 0,
            'last_update': None,
            'variables_count': 0
        }
        
        if DASHBOARD_AVAILABLE:
            self._create_app()
    
    def _create_app(self):
        """Create Dash application"""
        self.app = dash.Dash(__name__)
        
        # Layout
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🏭 Industrial Network Anomaly Detection", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px'}),
                html.H3("Real-time Zabbix Integration Monitor", 
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'margin': '10px'})
            ]),
            
            # Status bar
            html.Div(id='status-bar', 
                    style={'textAlign': 'center', 'padding': '15px', 
                           'backgroundColor': '#ecf0f1', 'margin': '10px', 
                           'borderRadius': '5px', 'fontSize': '16px'}),
            
            # Main chart
            html.Div([
                dcc.Graph(id='main-chart', style={'height': '500px'})
            ], style={'margin': '20px'}),
            
            # Anomaly chart
            html.Div([
                dcc.Graph(id='anomaly-chart', style={'height': '300px'})
            ], style={'margin': '20px'}),
            
            # Auto-refresh
            dcc.Interval(id='interval-component', interval=3000, n_intervals=0)
        ])
        
        # Callbacks
        @self.app.callback(
            [Output('status-bar', 'children'),
             Output('main-chart', 'figure'),
             Output('anomaly-chart', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            return self._get_dashboard_data()
    
    def _get_dashboard_data(self):
        """Get current dashboard data (thread-safe)"""
        with self.data_lock:
            # Status
            if self.stats['last_update']:
                time_diff = datetime.now() - self.stats['last_update']
                status_color = '#27ae60' if time_diff.seconds < 120 else '#e74c3c'
                status = html.Div([
                    html.Span(f"🔄 Cycles: {self.stats['cycles']} | ", 
                             style={'color': status_color, 'fontWeight': 'bold'}),
                    html.Span(f"⚠️ Anomalies: {self.stats['anomalies_detected']} | "),
                    html.Span(f"📊 Variables: {self.stats['variables_count']} | "),
                    html.Span(f"🕒 Last: {self.stats['last_update'].strftime('%H:%M:%S')}")
                ])
            else:
                status = "⏳ Waiting for data from Zabbix..."
            
            # Main data chart
            main_fig = go.Figure()
            if len(self.timestamps) > 0 and self.data_series:
                timestamps_list = list(self.timestamps)
                for var_name, values in self.data_series.items():
                    if len(values) > 0:
                        values_list = list(values)
                        min_len = min(len(timestamps_list), len(values_list))
                        if min_len > 0:
                            main_fig.add_trace(go.Scatter(
                                x=timestamps_list[-min_len:],
                                y=values_list[-min_len:],
                                mode='lines+markers',
                                name=var_name,
                                line=dict(width=2)
                            ))
            else:
                main_fig.add_annotation(
                    text="⏳ Waiting for real-time data...",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(size=18, color='#95a5a6')
                )
            
            main_fig.update_layout(
                title="📈 Real-time Industrial Network Data",
                xaxis_title="Time", yaxis_title="Value",
                hovermode='x unified', showlegend=True
            )
            
            # Anomaly chart
            anomaly_fig = go.Figure()
            if len(self.timestamps) > 0 and self.anomaly_series:
                timestamps_list = list(self.timestamps)
                for var_name, anomalies in self.anomaly_series.items():
                    if len(anomalies) > 0:
                        anomaly_list = list(anomalies)
                        min_len = min(len(timestamps_list), len(anomaly_list))
                        if min_len > 0:
                            colors = ['red' if a else 'green' for a in anomaly_list[-min_len:]]
                            anomaly_fig.add_trace(go.Scatter(
                                x=timestamps_list[-min_len:],
                                y=[1 if a else 0 for a in anomaly_list[-min_len:]],
                                mode='markers',
                                name=f'{var_name}',
                                marker=dict(color=colors, size=8)
                            ))
            else:
                anomaly_fig.add_annotation(
                    text="✅ No anomaly data yet",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(size=16, color='#27ae60')
                )
            
            anomaly_fig.update_layout(
                title="⚠️ Anomaly Detection Status",
                xaxis_title="Time", yaxis_title="Anomaly Status",
                showlegend=True
            )
            
            return status, main_fig, anomaly_fig
    
    def update_data(self, timestamp, data_dict, anomaly_dict):
        """Update dashboard data (thread-safe)"""
        if not DASHBOARD_AVAILABLE:
            return
        
        with self.data_lock:
            # Add timestamp
            self.timestamps.append(timestamp)
            
            # Add data points
            for var_name, value in data_dict.items():
                if var_name not in self.data_series:
                    self.data_series[var_name] = deque(maxlen=100)
                
                # Ensure value is valid
                if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                    self.data_series[var_name].append(float(value))
                else:
                    self.data_series[var_name].append(0.0)
            
            # Add anomaly flags
            for var_name, is_anomaly in anomaly_dict.items():
                if var_name not in self.anomaly_series:
                    self.anomaly_series[var_name] = deque(maxlen=100)
                self.anomaly_series[var_name].append(bool(is_anomaly))
            
            # Update stats
            self.stats['cycles'] += 1
            self.stats['anomalies_detected'] += sum(anomaly_dict.values())
            self.stats['last_update'] = timestamp
            self.stats['variables_count'] = len(data_dict)
    
    def start_server(self):
        """Start dashboard server"""
        if not DASHBOARD_AVAILABLE or self.is_running:
            return False
        
        def run_server():
            try:
                self.app.run_server(
                    host='127.0.0.1',
                    port=self.port,
                    debug=False,
                    use_reloader=False,
                    dev_tools_hot_reload=False
                )
            except Exception as e:
                print(f"Dashboard server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True
        time.sleep(3)  # Give server time to start
        return True


class SimplifiedZabbixMonitor:
    """Simplified Zabbix monitoring with reliable dashboard"""
    
    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # Core components
        self.zabbix_api = None
        self.anomaly_detector = SimpleAnomalyDetector()
        self.dashboard = SimpleDashboard(port=self.config['monitoring']['dashboard_port'])
        
        # Settings
        self.update_interval = self.config['data_collection']['update_interval']
        self.search_criteria = self.config['data_collection']['search_criteria']
        self.host_groups = self.config['data_collection']['host_groups']
        
        # Cache
        self.items_cache = None
    
    def _load_config(self, config_file):
        """Load configuration with defaults"""
        default_config = {
            "zabbix": {
                "url": "https://192.168.93.45",
                "user": "Admin",
                "password": "zabbix"
            },
            "data_collection": {
                "search_criteria": ["cpu", "memory", "bits", "temperature", "ping"],
                "host_groups": ["Zabbix servers"],
                "update_interval": 60
            },
            "monitoring": {
                "dashboard_port": 8052,
                "log_level": "INFO"
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Merge configs
                for section in default_config:
                    if section in user_config:
                        default_config[section].update(user_config[section])
                return default_config
        except FileNotFoundError:
            self.logger.info(f"Config {config_file} not found, using defaults")
            return default_config
        except Exception as e:
            self.logger.error(f"Config error: {e}")
            return default_config
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=getattr(logging, self.config['monitoring']['log_level'], logging.INFO),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('simplified_monitor.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def connect_zabbix(self) -> bool:
        """Connect to Zabbix API"""
        try:
            zabbix_config = self.config['zabbix']
            self.logger.info(f"🔗 Connecting to Zabbix: {zabbix_config['url']}")
            
            self.zabbix_api = ZabbixAPI(zabbix_config['url'])
            self.zabbix_api.session.verify = False
            self.zabbix_api.login(zabbix_config['user'], zabbix_config['password'])
            
            version = self.zabbix_api.apiinfo.version()
            self.logger.info(f"✅ Connected to Zabbix {version}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Zabbix connection failed: {e}")
            return False
    
    def discover_items(self) -> List[Dict]:
        """Discover monitoring items"""
        if self.items_cache:
            return self.items_cache
        
        try:
            # Get hosts
            all_hosts = []
            for group_name in self.host_groups:
                try:
                    groups = self.zabbix_api.hostgroup.get(filter={"name": group_name})
                    if groups:
                        hosts = self.zabbix_api.host.get(
                            groupids=[groups[0]['groupid']],
                            output=['hostid', 'host', 'name'],
                            filter={'status': 0}
                        )
                        all_hosts.extend(hosts)
                        self.logger.info(f"Found {len(hosts)} hosts in group '{group_name}'")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error with group '{group_name}': {e}")
            
            if not all_hosts:
                self.logger.error("❌ No hosts found")
                return []
            
            # Get items
            host_ids = [host['hostid'] for host in all_hosts]
            items = self.zabbix_api.item.get(
                hostids=host_ids,
                output=['itemid', 'name', 'key_', 'hostid'],
                monitored=True,
                filter={'value_type': [0, 3]}  # Numeric only
            )
            
            # Filter items
            host_lookup = {host['hostid']: host for host in all_hosts}
            filtered_items = []
            
            for item in items:
                item_name_lower = item['name'].lower()
                item_key_lower = item['key_'].lower()
                
                # Check search criteria
                for criteria in self.search_criteria:
                    if (criteria.lower() in item_name_lower or 
                        criteria.lower() in item_key_lower):
                        
                        if item['hostid'] in host_lookup:
                            item['host_name'] = host_lookup[item['hostid']]['host']
                            item['display_name'] = f"{item['host_name']}-{item['name']}"
                            filtered_items.append(item)
                        break
            
            self.items_cache = filtered_items
            self.logger.info(f"🔍 Found {len(filtered_items)} monitoring items")
            return filtered_items
            
        except Exception as e:
            self.logger.error(f"❌ Item discovery failed: {e}")
            return []
    
    def collect_data(self, items: List[Dict]) -> Optional[Dict[str, float]]:
        """Collect current data from Zabbix"""
        try:
            # Get recent data (last 10 minutes)
            time_to = int(time.time())
            time_from = time_to - 600  # 10 minutes
            
            item_ids = [item['itemid'] for item in items]
            
            # Get history
            history = self.zabbix_api.history.get(
                itemids=item_ids,
                time_from=time_from,
                time_till=time_to,
                output='extend',
                sortfield='clock',
                limit=500
            )
            
            # Process - get latest value for each item
            item_lookup = {item['itemid']: item for item in items}
            latest_data = {}
            
            for record in history:
                if record['itemid'] in item_lookup:
                    try:
                        item_name = item_lookup[record['itemid']]['display_name']
                        value = float(record['value'])
                        timestamp = int(record['clock'])
                        
                        # Keep latest value
                        if (item_name not in latest_data or 
                            timestamp > latest_data[item_name]['timestamp']):
                            latest_data[item_name] = {
                                'value': value,
                                'timestamp': timestamp
                            }
                    except (ValueError, TypeError):
                        continue
            
            # Extract values
            result = {name: data['value'] for name, data in latest_data.items()}
            
            if result:
                self.logger.info(f"📊 Collected {len(result)} data points")
                return result
            else:
                self.logger.warning("⚠️ No data collected")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Data collection failed: {e}")
            return None
    
    def run_monitoring(self):
        """Main monitoring loop"""
        self.logger.info("🏭 Starting simplified industrial network monitoring...")
        
        # Initialize
        if not self.connect_zabbix():
            return False
        
        items = self.discover_items()
        if not items:
            return False
        
        # Start dashboard
        dashboard_started = self.dashboard.start_server()
        if dashboard_started:
            self.logger.info(f"🌐 Dashboard: http://localhost:{self.config['monitoring']['dashboard_port']}")
        else:
            self.logger.warning("⚠️ Dashboard not available")
        
        # Main loop
        cycle = 0
        self.logger.info(f"⏰ Monitoring every {self.update_interval} seconds")
        
        while True:
            try:
                cycle += 1
                cycle_start = time.time()
                
                self.logger.info(f"🔄 Cycle #{cycle}")
                
                # Collect data
                data = self.collect_data(items)
                if data is None:
                    self.logger.warning(f"⚠️ Cycle #{cycle}: No data")
                    time.sleep(self.update_interval)
                    continue
                
                # Detect anomalies
                anomalies = self.anomaly_detector.detect_anomalies(data)
                
                # Update dashboard
                timestamp = datetime.now()
                self.dashboard.update_data(timestamp, data, anomalies)
                
                # Log results
                anomaly_count = sum(anomalies.values())
                self.logger.info(f"✅ Cycle #{cycle}: {len(data)} variables, {anomaly_count} anomalies")
                
                if anomaly_count > 0:
                    anomaly_vars = [var for var, is_anom in anomalies.items() if is_anom]
                    self.logger.warning(f"⚠️ ANOMALIES: {', '.join(anomaly_vars)}")
                
                # Wait for next cycle
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.update_interval - cycle_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in cycle #{cycle}: {e}")
                time.sleep(self.update_interval)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified Industrial Network Anomaly Detection')
    parser.add_argument('--config', '-c', default='config.json', help='Configuration file')
    parser.add_argument('--test', '-t', action='store_true', help='Test connections only')
    
    args = parser.parse_args()
    
    monitor = SimplifiedZabbixMonitor(args.config)
    
    if args.test:
        print("🧪 Testing system components...")
        print("=" * 50)
        
        # Test Zabbix
        print("1️⃣ Testing Zabbix connection...")
        zabbix_ok = monitor.connect_zabbix()
        
        if zabbix_ok:
            print("2️⃣ Testing item discovery...")
            items = monitor.discover_items()
            
            if items:
                print("3️⃣ Testing data collection...")
                data = monitor.collect_data(items)
                if data:
                    print(f"📊 Sample data: {len(data)} variables")
                    for var, val in list(data.items())[:3]:  # Show first 3
                        print(f"   {var}: {val}")
                    print("✅ All systems ready!")
                else:
                    print("⚠️ No data collected")
            else:
                print("⚠️ No items found")
        else:
            print("❌ Zabbix connection failed")
        
        print("=" * 50)
        return
    
    # Run monitoring
    print("🏭 Starting Simplified Industrial Network Monitor")
    print("   - Statistical anomaly detection")
    print("   - Real-time dashboard with working display")
    print("   - Press Ctrl+C to stop")
    print()
    
    monitor.run_monitoring()


if __name__ == "__main__":
    main()