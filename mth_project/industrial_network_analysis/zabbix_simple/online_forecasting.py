#!/usr/bin/env python3
"""
Online Forecasting Loop for Zabbix Integration
Real-time anomaly detection with clean stopping mechanism
"""

import json
import os
import sys
import time
import logging
import threading
import signal
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

# Add parent directory to import existing modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    from initial_model import get_initial_model, get_online_data
    from online_forecasting_multi_step import multistep_rolling_buffer_learning_prediction_with_dash
    from collect_data import ZabbixDataCollector
    from dash_plotter import DashRealTimePlotter
    print("✅ Imported existing modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure parent modules are available")
    sys.exit(1)

# Global stop flag for clean shutdown
stop_flag = threading.Event()


class ZabbixForecastingLoop:
    """Online forecasting loop with Zabbix integration"""
    
    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # Configuration
        self.context_length = self.config['model']['context_length']
        self.prediction_horizon = self.config['model']['prediction_horizon']
        self.model_path = self.config['model']['model_path']
        self.update_interval = self.config['data_collection']['update_interval']
        
        # Components
        self.model = None
        self.scalers = None
        self.variables = None
        self.collector = ZabbixDataCollector(config_file)
        self.monitoring_items = []
        self.dash_plotter = None
        
        # Data storage
        os.makedirs('temp_data', exist_ok=True)
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration"""
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=getattr(logging, self.config['monitoring']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('forecasting_loop.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for clean shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            stop_flag.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def setup_keyboard_listener(self):
        """Setup keyboard listener in separate thread"""
        def keyboard_listener():
            while not stop_flag.is_set():
                try:
                    user_input = input()
                    if user_input.upper() == 'Q':
                        self.logger.info("Quit command detected! Initiating graceful shutdown...")
                        stop_flag.set()
                        break
                except (EOFError, KeyboardInterrupt):
                    break
                except Exception:
                    pass
        
        listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
        listener_thread.start()
        return listener_thread
    
    def initialize_system(self) -> bool:
        """Initialize all system components"""
        try:
            self.logger.info("Initializing system components...")
            
            # Step 1: Connect to Zabbix
            print("Step 1: Connecting to Zabbix...")
            self.logger.info("Step 1: Connecting to Zabbix...")
            if not self.collector.connect():
                print("❌ Zabbix connection failed")
                self.logger.error("❌ Zabbix connection failed")
                return False
            print("✅ Zabbix connection successful")
            self.logger.info("✅ Zabbix connection successful")
            
            # Step 2: Check if model directory exists
            print(f"Step 2: Checking model directory: {self.model_path}")
            self.logger.info(f"Step 2: Checking model directory: {self.model_path}")
            if not os.path.exists(self.model_path):
                print(f"❌ Model directory not found: {self.model_path}")
                print("   Please train a model first using: python3 train_model.py --data your_data.csv")
                self.logger.error(f"❌ Model directory not found: {self.model_path}")
                self.logger.error("   Please train a model first using: python3 train_model.py --data your_data.csv")
                return False
            print("✅ Model directory exists")
            self.logger.info("✅ Model directory exists")
            
            # Step 3: Load trained model
            print("Step 3: Loading trained model...")
            self.logger.info("Step 3: Loading trained model...")
            try:
                self.model = get_initial_model(self.model_path)
                print("✅ Model loaded successfully")
                self.logger.info("✅ Model loaded successfully")
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
                print("   Check if model files exist and are valid")
                self.logger.error(f"❌ Model loading failed: {e}")
                self.logger.error("   Check if model files exist and are valid")
                return False
            
            # Step 4: Load online data and scalers
            print("Step 4: Loading online data and scalers...")
            self.logger.info("Step 4: Loading online data and scalers...")
            try:
                df_online, self.scalers, context_length, df_removed_nans_forecasting, df_removed_nans_classification, self.variables, model_mode = get_online_data(self.model_path)
                print("✅ Online data and scalers loaded")
                self.logger.info("✅ Online data and scalers loaded")
            except Exception as e:
                print(f"❌ Online data loading failed: {e}")
                print("   Check if training data files exist in model directory")
                self.logger.error(f"❌ Online data loading failed: {e}")
                self.logger.error("   Check if training data files exist in model directory")
                return False
            
            # Step 5: Verify parameters
            if context_length != self.context_length:
                self.logger.warning(f"⚠️ Context length mismatch: config={self.context_length}, model={context_length}")
                self.context_length = context_length
            
            # Step 5: Discover monitoring items
            print("Step 5: Discovering Zabbix monitoring items...")
            self.logger.info("Step 5: Discovering Zabbix monitoring items...")
            try:
                self.monitoring_items = self.collector.discover_items()
                if not self.monitoring_items:
                    print("❌ No monitoring items found")
                    print("   Check host groups and search criteria in config.json")
                    self.logger.error("❌ No monitoring items found")
                    self.logger.error("   Check host groups and search criteria in config.json")
                    return False
                print(f"✅ Found {len(self.monitoring_items)} monitoring items")
                self.logger.info(f"✅ Found {len(self.monitoring_items)} monitoring items")
            except Exception as e:
                print(f"❌ Item discovery failed: {e}")
                self.logger.error(f"❌ Item discovery failed: {e}")
                return False
            
            self.logger.info("🎉 System initialization completed successfully!")
            self.logger.info(f"   Model mode: {model_mode}")
            self.logger.info(f"   Context length: {context_length}")
            self.logger.info(f"   Variables: {len(self.variables)}")
            self.logger.info(f"   Monitoring items: {len(self.monitoring_items)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Unexpected error during system initialization: {e}")
            import traceback
            print(f"   Full traceback: {traceback.format_exc()}")
            self.logger.error(f"❌ Unexpected error during system initialization: {e}")
            self.logger.error(f"   Full traceback: {traceback.format_exc()}")
            return False
    
    def initialize_dashboard(self):
        """Initialize Dash dashboard (optional)"""
        try:
            dashboard_port = self.config['monitoring']['dashboard_port']
            self.logger.info(f"🌐 Initializing dashboard on port {dashboard_port}...")
            
            self.dash_plotter = DashRealTimePlotter()
            
            # Check if the DashRealTimePlotter has a start_server method with parameters
            if hasattr(self.dash_plotter, 'start_server'):
                # Use the original start_server method with proper parameters
                self.logger.info("📊 Starting dashboard server with network binding...")
                dashboard_thread = self.dash_plotter.start_server(
                    host='0.0.0.0',  # Bind to all interfaces for network access
                    port=dashboard_port,
                    debug=False
                )
                self.logger.info("✅ Dashboard started using original start_server method")
                
                # Give the server time to start
                time.sleep(3)
                
            else:
                # Fallback: start manually in thread (should not be needed with original class)
                def start_dashboard():
                    try:
                        self.dash_plotter.app.run_server(
                            host='0.0.0.0',  # Bind to all interfaces
                            port=dashboard_port,
                            debug=False,
                            threaded=True,
                            use_reloader=False
                        )
                    except Exception as e:
                        self.logger.error(f"Dashboard server failed to start: {e}")
                
                dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
                dashboard_thread.start()
                self.logger.info("✅ Dashboard started in fallback thread mode")
                time.sleep(3)
            
            # Get VM's IP address for connection info
            import socket
            hostname = socket.gethostname()
            try:
                local_ip = socket.gethostbyname(hostname)
            except:
                local_ip = "localhost"
            
            self.logger.info(f"📡 Dashboard accessible at:")
            self.logger.info(f"   Local: http://localhost:{dashboard_port}")
            self.logger.info(f"   Network: http://{local_ip}:{dashboard_port}")
            print(f"🌐 Dashboard accessible from Windows VM at: http://{local_ip}:{dashboard_port}")
            print(f"📊 Dashboard should now be running and accessible!")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Dashboard initialization failed: {e}")
            self.logger.warning("   Continuing without dashboard...")
            self.dash_plotter = None
    
    def collect_and_prepare_data(self) -> Optional[pd.DataFrame]:
        """Collect current data and prepare for prediction"""
        try:
            # Collect recent data from Zabbix
            raw_data = self.collector.collect_recent_data(self.monitoring_items, hours_back=2)
            
            if raw_data is None or raw_data.empty:
                self.logger.warning("No data collected from Zabbix")
                return None
            
            # Ensure we have enough data for context
            if len(raw_data) < self.context_length:
                self.logger.warning(f"Insufficient data: need {self.context_length}, have {len(raw_data)}")
                return None
            
            # Match columns to model variables (simplified approach)
            available_columns = raw_data.columns.tolist()
            
            if len(available_columns) >= len(self.variables):
                # Use first N columns matching model size
                selected_data = raw_data[available_columns[:len(self.variables)]].copy()
                selected_data.columns = self.variables
            else:
                # Pad with zeros if not enough columns
                selected_data = pd.DataFrame(index=raw_data.index, columns=self.variables)
                for i, col in enumerate(available_columns):
                    if i < len(self.variables):
                        selected_data[self.variables[i]] = raw_data[col]
                selected_data = selected_data.fillna(0)
            
            # Apply differencing (like training)
            prepared_data = selected_data.diff().dropna()
            
            # Get recent data for prediction
            recent_data = prepared_data.tail(self.context_length * 2)
            
            return recent_data
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            return None
    
    def run_prediction_cycle(self, df: pd.DataFrame, cycle: int) -> bool:
        """Run single prediction cycle"""
        try:
            # Save temporary data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = f"temp_data/cycle_{cycle:04d}_{timestamp}.csv"
            df.to_csv(temp_file)
            
            # Run forecasting
            predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df = multistep_rolling_buffer_learning_prediction_with_dash(
                initial_model=self.model,
                df_online=df,
                scalers=self.scalers,
                context_length=self.context_length,
                df_removed_nans_forecasting=df,  # Simplified
                df_removed_nans_classification=df,  # Simplified
                dash_plotter=self.dash_plotter,  # Enable dashboard
                variables=self.variables,
                prediction_horizon=self.prediction_horizon
            )
            
            # Log results
            if not predictions_df.empty:
                mae = np.mean(np.abs(predictions_df.values - actuals_df.values)) if not actuals_df.empty else 0
                self.logger.info(f"Cycle {cycle}: Generated {len(predictions_df)} predictions, MAE: {mae:.6f}")
            else:
                self.logger.warning(f"Cycle {cycle}: No predictions generated")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Prediction cycle {cycle} failed: {e}")
            return False
    
    def run_monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info("🏭 Starting real-time forecasting loop...")
        self.logger.info("Press 'Q' + Enter to stop gracefully")
        
        # Setup signal handlers and keyboard listener
        self.setup_signal_handlers()
        keyboard_thread = self.setup_keyboard_listener()
        
        cycle = 0
        start_time = time.time()
        
        try:
            while not stop_flag.is_set():
                cycle += 1
                cycle_start = time.time()
                
                self.logger.info(f"🔄 Cycle #{cycle} started")
                
                # Collect and prepare data
                prepared_data = self.collect_and_prepare_data()
                
                if prepared_data is None:
                    self.logger.warning(f"Cycle #{cycle}: Skipping due to data issues")
                else:
                    # Run prediction
                    success = self.run_prediction_cycle(prepared_data, cycle)
                    
                    if success:
                        self.logger.info(f"✅ Cycle #{cycle} completed")
                    else:
                        self.logger.warning(f"⚠️ Cycle #{cycle} had issues")
                
                # Wait for next cycle or check stop flag
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.update_interval - cycle_time)
                
                # Interruptible sleep
                end_time = time.time() + sleep_time
                while time.time() < end_time and not stop_flag.is_set():
                    time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Unexpected error in monitoring loop: {e}")
        
        finally:
            # Clean shutdown
            total_time = time.time() - start_time
            self.logger.info("🛑 Graceful shutdown initiated")
            self.logger.info(f"📊 Session summary:")
            self.logger.info(f"   Total cycles: {cycle}")
            self.logger.info(f"   Total runtime: {total_time:.1f} seconds")
            self.logger.info(f"   Average cycle time: {total_time/max(cycle,1):.1f} seconds")
            self.logger.info("✅ Shutdown completed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Zabbix Real-time Forecasting Loop')
    parser.add_argument('--test', action='store_true', help='Test system components')
    args = parser.parse_args()
    
    forecaster = ZabbixForecastingLoop()
    
    if args.test:
        print("🧪 Testing system components...")
        print("=" * 50)
        
        try:
            success = forecaster.initialize_system()
            if success:
                print("=" * 50)
                print("✅ All systems ready for monitoring!")
                print(f"   Found {len(forecaster.monitoring_items)} monitoring items")
                print(f"   Model variables: {len(forecaster.variables)}")
                print(f"   Context length: {forecaster.context_length}")
            else:
                print("=" * 50)
                print("❌ System initialization failed - check logs above for details")
                return 1
        except Exception as e:
            print(f"❌ Unexpected error during testing: {e}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            return 1
        
        return 0
    
    # Initialize and run monitoring
    print("🏭 Zabbix Real-time Anomaly Detection")
    print("=" * 40)
    print("Initializing system...")
    
    if not forecaster.initialize_system():
        print("❌ System initialization failed")
        return 1
    
    print("✅ System ready!")
    
    # Initialize dashboard (optional)
    forecaster.initialize_dashboard()
    if forecaster.dash_plotter:
        print(f"🌐 Dashboard: http://localhost:{forecaster.config['monitoring']['dashboard_port']}")
    
    print("Starting monitoring loop...")
    print("Press 'Q' + Enter to stop gracefully")
    print("=" * 40)
    
    try:
        forecaster.run_monitoring_loop()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    return 0


if __name__ == "__main__":
    exit(main())
    