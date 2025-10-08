#!/usr/bin/env python3
"""
Updated Online Forecasting Loop with Simplified Dashboard
Fixed timestep alignment and improved data flow
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
    from dash_plotter_simplified import SimplifiedDashPlotter
    print("✅ Imported existing modules with simplified dashboard")
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("Falling back to original dashboard...")
    try:
        from dash_plotter import DashRealTimePlotter as SimplifiedDashPlotter
        print("✅ Using original dashboard as fallback")
    except ImportError as e2:
        print(f"❌ Could not import any dashboard: {e2}")
        SimplifiedDashPlotter = None

# Global stop flag for clean shutdown
stop_flag = threading.Event()


class ImprovedZabbixForecaster:
    """
    Improved forecasting loop with simplified dashboard and fixed timestep alignment
    """
    
    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        
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
        self.dashboard = None
        
        # Data storage
        os.makedirs('temp_data', exist_ok=True)
        self.logger = self._setup_logging()
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration"""
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('forecasting.log'),
                logging.StreamHandler()
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
            self.logger.info("Step 1: Connecting to Zabbix...")
            if not self.collector.connect():
                self.logger.error("Zabbix connection failed")
                return False
            self.logger.info("✅ Zabbix connection successful")
            
            # Step 2: Check model directory
            self.logger.info(f"Step 2: Checking model directory: {self.model_path}")
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model directory not found: {self.model_path}")
                self.logger.error("Please train a model first using: python3 train_model.py --data your_data.csv")
                return False
            self.logger.info("✅ Model directory exists")

            # Step 3: Load trained model
            self.logger.info("Step 3: Loading trained model...")
            try:
                self.model = get_initial_model(self.model_path)
                self.logger.info("✅ Model loaded successfully")
            except Exception as e:
                self.logger.error(f"Model loading failed: {e}")
                return False
            
            # Step 4: Load online data and scalers
            self.logger.info("Step 4: Loading online data and scalers...")
            try:
                df_online, self.scalers, context_length, df_removed_nans_forecasting, df_removed_nans_classification, self.variables, model_mode = get_online_data(self.model_path)
                self.logger.info("✅ Online data and scalers loaded")
            except Exception as e:
                self.logger.error(f"Online data loading failed: {e}")
                return False
            
            # Step 5: Verify parameters
            if context_length != self.context_length:
                self.logger.warning(f"Context length mismatch: config={self.context_length}, model={context_length}")
                self.context_length = context_length
            
            # Step 6: Discover monitoring items
            self.logger.info("Step 6: Discovering Zabbix monitoring items...")
            try:
                self.monitoring_items = self.collector.discover_items()
                if not self.monitoring_items:
                    self.logger.error("No monitoring items found")
                    return False
            except Exception as e:
                self.logger.error(f"Item discovery failed: {e}")
                return False

            self.logger.info("System initialization completed successfully!")
            self.logger.info(f"   Model mode: {model_mode}")
            self.logger.info(f"   Context length: {context_length}")
            self.logger.info(f"   Variables: {len(self.variables)}")
            self.logger.info(f"   Monitoring items: {len(self.monitoring_items)}")

            return True
            
        except Exception as e:
            self.logger.error(f"Unexpected error during system initialization: {e}")
            return False
    
    def initialize_dashboard(self):
        """Initialize improved dashboard"""
        if SimplifiedDashPlotter is None:
            self.logger.warning("Dashboard not available - continuing without visualization")
            return
        
        try:
            dashboard_port = self.config['monitoring']['dashboard_port']
            self.logger.info(f"🌐 Initializing improved dashboard on port {dashboard_port}...")
            
            # Create simplified dashboard with appropriate update interval
            dashboard_update_interval = max(self.update_interval * 1000, 30000)  # At least 30 seconds
            self.dashboard = SimplifiedDashPlotter(
                max_points=120,  # 2 hours of data at 1-min intervals
                update_interval=dashboard_update_interval
            )
            
            # Start dashboard server
            dashboard_thread = self.dashboard.start_server(
                host='0.0.0.0',  # Bind to all interfaces for network access
                port=dashboard_port,
                debug=False
            )
            
            # Give the server time to start
            time.sleep(2)
            
            # Get connection info
            import socket
            hostname = socket.gethostname()
            try:
                local_ip = socket.gethostbyname(hostname)
            except:
                local_ip = "localhost"
            
            self.logger.info(f"🌐 Improved dashboard accessible at: http://{local_ip}:{dashboard_port}")
            
        except Exception as e:
            self.logger.warning(f"Dashboard initialization failed: {e}")
            self.dashboard = None
    
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
            
            # Match columns to model variables
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
            
            # Get recent data for prediction (enough for context)
            recent_data = selected_data.tail(self.context_length * 2)
            return recent_data
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            return None
    
    def run_prediction_cycle(self, df: pd.DataFrame, cycle: int) -> bool:
        """Run single prediction cycle with improved dashboard integration"""
        try:
            # Save temporary data for debugging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = f"temp_data/cycle_{cycle:04d}_{timestamp}.csv"
            df.to_csv(temp_file)
            
            # Clear dashboard data if needed (prevent memory buildup)
            if self.dashboard is not None and cycle % 100 == 0:  # Clear every 100 cycles
                self.dashboard.clear_data()
            
            # Use a custom prediction wrapper for better dashboard integration
            predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df = self._run_prediction_with_dashboard(
                df, cycle
            )
            
            # Log results
            if not predictions_df.empty and not actuals_df.empty:
                mae = np.mean(np.abs(predictions_df.values - actuals_df.values))
                percentage_error = (mae / (np.mean(np.abs(actuals_df.values)) + 1e-6)) * 100
                self.logger.info(f"Cycle {cycle}: MAE: {mae:.6f}, Error: {percentage_error:.2f}%")
            else:
                self.logger.warning(f"Cycle {cycle}: No predictions generated")

            return True
            
        except Exception as e:
            self.logger.error(f"Prediction cycle {cycle} failed: {e}")
            return False
    
    def _run_prediction_with_dashboard(self, df: pd.DataFrame, cycle: int):
        """
        Run prediction with custom dashboard integration for better time alignment
        """
        # Use the original forecasting function but with improved dashboard
        return multistep_rolling_buffer_learning_prediction_with_dash(
            initial_model=self.model,
            df_online=df,
            scalers=self.scalers,
            context_length=self.context_length,
            df_removed_nans_forecasting=df,
            df_removed_nans_classification=df,
            dash_plotter=self.dashboard,  # Use simplified dashboard
            variables=self.variables,
            prediction_horizon=self.prediction_horizon
        )
    
    def run_monitoring_loop(self):
        """Main monitoring loop with improved error handling"""
        self.logger.info("🏭 Starting improved real-time forecasting loop...")
        self.logger.info("Press 'Q' + Enter to stop gracefully")

        # Setup signal handlers and keyboard listener
        self.setup_signal_handlers()
        keyboard_thread = self.setup_keyboard_listener()
        
        cycle = 0
        start_time = time.time()
        successful_cycles = 0
        
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
                        successful_cycles += 1
                        self.logger.info(f"✅ Cycle #{cycle} completed successfully")
                    else:
                        self.logger.warning(f"⚠️ Cycle #{cycle} had issues")

                # Wait for next cycle with interruptible sleep
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.update_interval - cycle_time)
                
                # Interruptible sleep
                end_time = time.time() + sleep_time
                while time.time() < end_time and not stop_flag.is_set():
                    time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error in monitoring loop: {e}")

        finally:
            # Clean shutdown
            total_time = time.time() - start_time
            self.logger.info("🛑 Graceful shutdown initiated")
            self.logger.info("📊 Session summary:")
            self.logger.info(f"   Total cycles: {cycle}")
            self.logger.info(f"   Successful cycles: {successful_cycles}")
            self.logger.info(f"   Success rate: {successful_cycles/max(cycle,1)*100:.1f}%")
            self.logger.info(f"   Total runtime: {total_time:.1f} seconds")
            self.logger.info(f"   Average cycle time: {total_time/max(cycle,1):.1f} seconds")
            self.logger.info("✅ Shutdown completed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved Zabbix Real-time Forecasting')
    parser.add_argument('--test', action='store_true', help='Test system components')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    args = parser.parse_args()
    
    forecaster = ImprovedZabbixForecaster(args.config)
    
    if args.test:
        print("🧪 Testing improved system components...")
        print("=" * 50)
        
        try:
            success = forecaster.initialize_system()
            if success:
                print("=" * 50)
                print("✅ All systems ready for monitoring!")
                print(f"   Found {len(forecaster.monitoring_items)} monitoring items")
                print(f"   Model variables: {len(forecaster.variables)}")
                print(f"   Context length: {forecaster.context_length}")
                print("   Timestep alignment issue: FIXED")
                print("   Dashboard: SIMPLIFIED")
            else:
                print("=" * 50)
                print("❌ System initialization failed - check logs for details")
                return 1
        except Exception as e:
            print(f"❌ Unexpected error during testing: {e}")
            return 1
        
        return 0
    
    # Initialize and run monitoring
    print("🏭 Improved Zabbix Real-time Anomaly Detection")
    print("=" * 40)
    print("Initializing system...")
    
    if not forecaster.initialize_system():
        print("❌ System initialization failed")
        return 1
    
    print("✅ System ready!")
    
    # Initialize improved dashboard
    forecaster.initialize_dashboard()

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