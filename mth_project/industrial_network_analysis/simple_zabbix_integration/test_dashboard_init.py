#!/usr/bin/env python3
"""
🧪 Dashboard Initialization Test

Quick test to check if dashboard initializes properly.
"""

import sys
import os
import json
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dashboard_initialization():
    """Test dashboard initialization"""
    print("🧪 Testing Dashboard Initialization")
    print("=" * 40)
    
    try:
        # Test import
        print("1. Testing import...")
        from standalone_dashboard import StandaloneDashboard
        print("✅ Import successful")
        
        # Test config loading
        print("2. Testing config...")
        with open('config.json', 'r') as f:
            config = json.load(f)
        dashboard_port = config['monitoring']['dashboard_port']
        print(f"✅ Config loaded, dashboard port: {dashboard_port}")
        
        # Test dashboard creation
        print("3. Testing dashboard creation...")
        dashboard = StandaloneDashboard(port=dashboard_port, debug=False)
        print("✅ Dashboard instance created")
        
        # Test server start (briefly)
        print("4. Testing server start...")
        dashboard.start_server()
        print(f"✅ Dashboard server started at http://localhost:{dashboard_port}")
        
        # Test data update
        print("5. Testing data update...")
        from datetime import datetime
        timestamp = datetime.now()
        predictions = {'CPU_Usage': 45.2, 'Memory': 67.8}
        dashboard.update_data(timestamp, predictions, is_anomaly=False)
        print("✅ Dashboard data update successful")
        
        print("\n🎉 All tests passed!")
        print(f"📊 Dashboard should be accessible at: http://localhost:{dashboard_port}")
        print("🔧 Dashboard initialization should work in main program")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_dashboard_initialization()
    if success:
        print("\n✅ Dashboard is ready for main_forecasting.py!")
        print("   The 'dash_plotter is None' issue should be resolved.")
    else:
        print("\n❌ Dashboard has issues that need to be fixed.")
    
    input("\nPress Enter to exit and stop dashboard server...")