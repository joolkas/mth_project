#!/usr/bin/env python3
"""
🔍 Debug Version of Zabbix Monitor

This version has extra logging and error handling to help diagnose service issues.
"""

import sys
import os
import traceback
import time

# Add verbose logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/opt/anomaly_detection/debug.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function with extensive error handling"""
    try:
        logger.info("🚀 DEBUG: Starting Zabbix monitor debug version")
        logger.info(f"🐍 DEBUG: Python version: {sys.version}")
        logger.info(f"📁 DEBUG: Working directory: {os.getcwd()}")
        logger.info(f"🛤️ DEBUG: Python path: {sys.path}")
        
        # Check if config exists
        config_file = 'config.json'
        if not os.path.exists(config_file):
            logger.error(f"❌ Config file not found: {config_file}")
            return 1
        
        logger.info(f"✅ Config file found: {config_file}")
        
        # Try to import the main monitor class
        logger.info("📦 DEBUG: Attempting imports...")
        
        # Add parent directory to path for imports
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        sys.path.insert(0, parent_dir)
        logger.info(f"📁 DEBUG: Added to path: {parent_dir}")
        
        try:
            from zabbix_monitor import ZabbixAnomalyMonitor
            logger.info("✅ Successfully imported ZabbixAnomalyMonitor")
        except ImportError as e:
            logger.error(f"❌ Import error: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return 1
        
        # Create monitor instance
        logger.info("🔧 DEBUG: Creating monitor instance...")
        monitor = ZabbixAnomalyMonitor(config_file)
        logger.info("✅ Monitor instance created")
        
        # Run the monitoring loop
        logger.info("🔄 DEBUG: Starting monitoring loop...")
        monitor.run_monitoring_loop()
        
    except KeyboardInterrupt:
        logger.info("🛑 DEBUG: Stopped by user (Ctrl+C)")
        return 0
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())