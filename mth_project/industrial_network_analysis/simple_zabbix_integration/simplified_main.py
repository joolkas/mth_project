#!/usr/bin/env python3
"""
🏭 Simplified Industrial Network Anomaly Detection with Zabbix Integration

A streamlined version focusing on core functionality:
- Real-time data collection from Zabbix
- Simple anomaly detection using statistical methods
- Live dashboard with working display
- Minimal dependencies and clear architecture

Author: Industrial Network Analysis System
Version: 2.0 (Simplified)
"""

import sys
import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from pyzabbix import ZabbixAPI
import urllib3

# Dashboard imports
try:
    import dash
    from dash import dcc, html, Input, Output
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("⚠️ Dashboard dependencies not available. Install with: pip install dash plotly")

# Disable warnings
urllib3.disable_warnings()
warnings.filterwarnings('ignore')


class SimpleAnomalyDetector:
    """Simple statistical anomaly detection"""
    
    def __init__(self, window_size=30, threshold_multiplier=2.5):
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self.data_history = {}
        
    def update_and_detect(self, data_dict: Dict[str, float]) -> Dict[str, bool]:
        """Update history and detect anomalies"""
        anomalies = {}
        
        for variable, value in data_dict.items():
            # Initialize history if needed
            if variable not in self.data_history:
                self.data_history[variable] = []
            
            # Add new value
            self.data_history[variable].append(value)
            
            # Keep only recent history
            if len(self.data_history[variable]) > self.window_size:
                self.data_history[variable] = self.data_history[variable][-self.window_size:]
            
            # Detect anomaly (need at least 10 points)
            if len(self.data_history[variable]) >= 10:
                recent_data = np.array(self.data_history[variable])
                mean_val = np.mean(recent_data[:-1])  # Exclude current point
                std_val = np.std(recent_data[:-1])
                
                # Z-score anomaly detection
                if std_val > 0:
                    z_score = abs((value - mean_val) / std_val)
                    anomalies[variable] = z_score > self.threshold_multiplier
                else:
                    anomalies[variable] = False
            else:
                anomalies[variable] = False
                
        return anomalies


class SimpleDashboard:
    """Simplified dashboard with working display"""
    
    def __init__(self, port=8052):
        self.port = port
        self.app = None
        self.server_thread = None
        self.is_running = False
        
        # Data storage
        self.max_points = 100
        self.timestamps = []
        self.data_points = {}
        self.anomaly_flags = {}
        self.system_stats = {
            'cycles': 0,
            'anomalies': 0,
            'last_update': None,
            'variables': set()
        }
        
        # Thread lock for data safety
        self.data_lock = threading.Lock()
        
        if DASHBOARD_AVAILABLE:
            self._setup_app()
    
    def _setup_app(self):
        """Setup Dash application"""
        self.app = dash.Dash(__name__)
        
        # Simple layout
        self.app.layout = html.Div([
            html.H1("🏭 Industrial Network Monitor", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # Status row
            html.Div([
                html.Div(id='status-info', 
                        style={'textAlign': 'center', 'padding': '10px',
                               'background': '#ecf0f1', 'margin': '10px'})
            ]),
            
            # Main chart
            html.Div([
                dcc.Graph(id='live-chart', style={'height': '400px'})
            ]),
            
            # Anomaly chart
            html.Div([
                dcc.Graph(id='anomaly-chart', style={'height': '300px'})
            ]),
            
            # Auto refresh
            dcc.Interval(id='interval', interval=2000, n_intervals=0)
        ])
        
        # Callbacks
        @self.app.callback(
            [Output('status-info', 'children'),
             Output('live-chart', 'figure'),
             Output('anomaly-chart', 'figure')],
            [Input('interval', 'n_intervals')]
        )
        def update_dashboard(n):
            return self._update_display()
    
    def _update_display(self):
        """Update all dashboard components"""
        with self.data_lock:
            # Status info
            if self.system_stats['last_update']:
                time_diff = datetime.now() - self.system_stats['last_update']
                status_color = 'green' if time_diff.seconds < 120 else 'red'
                status_text = html.Div([
                    html.Span(f"🔄 Cycles: {self.system_stats['cycles']} | ", 
                             style={'color': status_color}),
                    html.Span(f"⚠️ Anomalies: {self.system_stats['anomalies']} | "),
                    html.Span(f"📊 Variables: {len(self.system_stats['variables'])} | "),
                    html.Span(f"🕒 Last: {self.system_stats['last_update'].strftime('%H:%M:%S')}")
                ])
            else:
                status_text = "⏳ Waiting for data..."
            
            # Live data chart
            live_fig = go.Figure()
            if len(self.timestamps) > 0:
                for var_name, values in self.data_points.items():
                    if len(values) > 0:
                        live_fig.add_trace(go.Scatter(
                            x=self.timestamps[-len(values):],
                            y=values,
                            mode='lines+markers',
                            name=var_name,
                            line=dict(width=2)
                        ))
            
            live_fig.update_layout(
                title="📈 Real-time Data",
                xaxis_title="Time",
                yaxis_title="Value",
                height=400
            )
            
            # Anomaly chart
            anomaly_fig = go.Figure()
            if len(self.timestamps) > 0:
                for var_name, flags in self.anomaly_flags.items():
                    if len(flags) > 0:
                        # Create anomaly indicators
                        y_vals = [1 if flag else 0 for flag in flags]
                        colors = ['red' if flag else 'green' for flag in flags]
                        
                        anomaly_fig.add_trace(go.Scatter(
                            x=self.timestamps[-len(flags):],
                            y=y_vals,
                            mode='markers',
                            name=f'{var_name} Anomalies',
                            marker=dict(color=colors, size=8)
                        ))
            
            anomaly_fig.update_layout(
                title="⚠️ Anomaly Detection",
                xaxis_title="Time",
                yaxis_title="Anomaly",
                height=300
            )
            
            return status_text, live_fig, anomaly_fig
    
    def update_data(self, timestamp, data_dict, anomaly_dict):
        """Update dashboard data safely"""
        if not DASHBOARD_AVAILABLE:
            return
            
        with self.data_lock:
            # Add timestamp
            self.timestamps.append(timestamp)
            if len(self.timestamps) > self.max_points:
                self.timestamps = self.timestamps[-self.max_points:]
            
            # Add data points
            for var_name, value in data_dict.items():
                if var_name not in self.data_points:
                    self.data_points[var_name] = []
                self.data_points[var_name].append(value)
                if len(self.data_points[var_name]) > self.max_points:
                    self.data_points[var_name] = self.data_points[var_name][-self.max_points:]
            
            # Add anomaly flags
            for var_name, is_anomaly in anomaly_dict.items():
                if var_name not in self.anomaly_flags:
                    self.anomaly_flags[var_name] = []
                self.anomaly_flags[var_name].append(is_anomaly)
                if len(self.anomaly_flags[var_name]) > self.max_points:
                    self.anomaly_flags[var_name] = self.anomaly_flags[var_name][-self.max_points:]
            
            # Update stats
            self.system_stats['cycles'] += 1
            self.system_stats['anomalies'] += sum(anomaly_dict.values())
            self.system_stats['last_update'] = timestamp
            self.system_stats['variables'].update(data_dict.keys())
    
    def start_server(self):
        """Start dashboard server"""
        if not DASHBOARD_AVAILABLE or self.is_running:
            return
        
        def run_server():
            try:
                self.app.run_server(
                    host='0.0.0.0',
                    port=self.port,
                    debug=False,
                    use_reloader=False
                )
            except Exception as e:
                print(f"❌ Dashboard server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True
        time.sleep(2)
        print(f"✅ Dashboard started at http://localhost:{self.port}")


class SimplifiedZabbixMonitor:
    """Simplified Zabbix monitoring system"""
    
    def __init__(self, config_file="simplified_config.json"):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # Components
        self.zabbix_api = None
        self.anomaly_detector = SimpleAnomalyDetector()
        self.dashboard = SimpleDashboard(port=self.config['monitoring']['dashboard_port'])
        
        # Data collection settings
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
                "search_criteria": ["cpu", "memory", "bits", "temperature"],
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
                # Merge with defaults
                for key in default_config:
                    if key in user_config:
                        if isinstance(default_config[key], dict):
                            default_config[key].update(user_config[key])
                        else:
                            default_config[key] = user_config[key]
                return default_config
        except FileNotFoundError:
            self.logger.info(f"Config file {config_file} not found, using defaults and Zabbix config")
            # Try to load from the original config
            try:
                with open("config.json", 'r') as f:
                    original_config = json.load(f)
                    if 'zabbix' in original_config:
                        default_config['zabbix'] = original_config['zabbix']
            except:
                pass
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
        """Connect to Zabbix"""
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
            
            # Filter and format items
            host_lookup = {host['hostid']: host for host in all_hosts}
            filtered_items = []
            
            for item in items:
                item_name_lower = item['name'].lower()
                item_key_lower = item['key_'].lower()
                
                # Check if matches search criteria
                for criteria in self.search_criteria:
                    if (criteria.lower() in item_name_lower or 
                        criteria.lower() in item_key_lower):
                        
                        if item['hostid'] in host_lookup:
                            item['host_name'] = host_lookup[item['hostid']]['host']
                            item['display_name'] = f"{item['host_name']}_{item['name']}"
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
            time_from = time_to - 600  # 10 minutes back
            
            item_ids = [item['itemid'] for item in items]
            
            # Get history
            history = self.zabbix_api.history.get(
                itemids=item_ids,
                time_from=time_from,
                time_till=time_to,
                output='extend',
                sortfield='clock',
                limit=1000  # Limit results
            )
            
            # Process data - get latest value for each item
            item_lookup = {item['itemid']: item for item in items}
            latest_data = {}
            
            for record in history:
                if record['itemid'] in item_lookup:
                    try:
                        item_name = item_lookup[record['itemid']]['display_name']
                        value = float(record['value'])
                        timestamp = int(record['clock'])
                        
                        # Keep only the latest value for each item
                        if (item_name not in latest_data or 
                            timestamp > latest_data[item_name]['timestamp']):
                            latest_data[item_name] = {
                                'value': value,
                                'timestamp': timestamp
                            }
                    except (ValueError, TypeError):
                        continue
            
            # Extract just the values
            result = {name: data['value'] for name, data in latest_data.items()}
            
            if result:
                self.logger.info(f"📊 Collected data for {len(result)} variables")
                return result
            else:
                self.logger.warning("⚠️ No data collected")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Data collection failed: {e}")
            return None
    
    def run_monitoring(self):
        """Main monitoring loop"""
        self.logger.info("🏭 Starting simplified monitoring...")
        
        # Initialize connections
        if not self.connect_zabbix():
            return False
        
        # Discover items
        items = self.discover_items()
        if not items:
            return False
        
        # Start dashboard
        self.dashboard.start_server()
        
        # Main loop
        cycle = 0
        self.logger.info(f"⏰ Starting monitoring cycles (every {self.update_interval} seconds)")
        self.logger.info(f"🌐 Dashboard: http://localhost:{self.config['monitoring']['dashboard_port']}")
        
        while True:
            try:
                cycle += 1
                cycle_start = time.time()
                
                self.logger.info(f"🔄 Cycle #{cycle}")
                
                # Collect data
                data = self.collect_data(items)
                if data is None:
                    self.logger.warning(f"⚠️ Cycle #{cycle}: No data, skipping")
                    time.sleep(self.update_interval)
                    continue
                
                # Detect anomalies
                anomalies = self.anomaly_detector.update_and_detect(data)
                
                # Update dashboard
                timestamp = datetime.now()
                self.dashboard.update_data(timestamp, data, anomalies)
                
                # Log results
                anomaly_count = sum(anomalies.values())
                self.logger.info(f"✅ Cycle #{cycle}: {len(data)} variables, {anomaly_count} anomalies")
                
                if anomaly_count > 0:
                    anomaly_vars = [var for var, is_anom in anomalies.items() if is_anom]
                    self.logger.warning(f"⚠️ Anomalies detected in: {', '.join(anomaly_vars)}")
                
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


def test_dashboard_only():
    """Test dashboard with simulated data"""
    print("🧪 Testing dashboard with simulated data...")
    
    dashboard = SimpleDashboard(port=8053)
    dashboard.start_server()
    
    print("📊 Generating test data...")
    try:
        for i in range(100):
            timestamp = datetime.now() - timedelta(minutes=100-i)
            
            # Simulate some data with occasional anomalies
            data = {
                'CPU_Usage': 50 + 30 * np.sin(i * 0.1) + np.random.normal(0, 5),
                'Memory_Usage': 60 + 20 * np.cos(i * 0.15) + np.random.normal(0, 3),
                'Network_Bits': 1000 + 500 * np.sin(i * 0.2) + np.random.normal(0, 100)
            }
            
            # Add occasional anomalies
            if i % 20 == 0:
                data['CPU_Usage'] += 50  # Spike
            
            # Simple anomaly detection
            anomalies = {}
            for var, val in data.items():
                anomalies[var] = abs(val) > 100  # Simple threshold
            
            dashboard.update_data(timestamp, data, anomalies)
            time.sleep(0.1)
        
        print(f"✅ Dashboard test running at http://localhost:8053")
        print("Press Ctrl+C to stop...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Test stopped")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified Industrial Network Anomaly Detection')
    parser.add_argument('--config', '-c', default='simplified_config.json', help='Configuration file')
    parser.add_argument('--test-dashboard', '-td', action='store_true', help='Test dashboard only')
    parser.add_argument('--test', '-t', action='store_true', help='Test connections only')
    
    args = parser.parse_args()
    
    if args.test_dashboard:
        test_dashboard_only()
        return
    
    monitor = SimplifiedZabbixMonitor(args.config)
    
    if args.test:
        print("🧪 Testing system components...")
        print("=" * 50)
        
        # Test Zabbix connection
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
                    print("✅ All systems ready!")
                else:
                    print("⚠️ Data collection failed")
            else:
                print("⚠️ No items found")
        else:
            print("❌ Zabbix connection failed")
        
        print("=" * 50)
        return
    
    # Run monitoring
    print("🏭 Starting Simplified Industrial Network Monitor")
    print("   - Lightweight anomaly detection")
    print("   - Real-time dashboard")
    print("   - Press Ctrl+C to stop")
    print()
    
    monitor.run_monitoring()


if __name__ == "__main__":
    main()