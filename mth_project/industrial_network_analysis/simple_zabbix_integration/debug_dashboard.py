#!/usr/bin/env python3
"""
🔍 Debug Dashboard Import Issues

Let's find out what's really preventing the dashboard from working.
"""

import sys
import os
import traceback

def debug_dashboard_import():
    """Debug dashboard import step by step"""
    print("🔍 Debugging Dashboard Import Issues")
    print("=" * 50)
    
    try:
        print("1. Testing basic Python imports...")
        import json
        import datetime
        print("✅ Basic imports work")
        
        print("2. Testing dashboard dependencies...")
        try:
            import dash
            print(f"✅ Dash {dash.__version__} available")
        except ImportError as e:
            print(f"❌ Dash import failed: {e}")
            return False
            
        try:
            import plotly
            print(f"✅ Plotly {plotly.__version__} available")
        except ImportError as e:
            print(f"❌ Plotly import failed: {e}")
            return False
        
        print("3. Testing config loading...")
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            dashboard_port = config['monitoring']['dashboard_port']
            print(f"✅ Config loaded, dashboard port: {dashboard_port}")
        except Exception as e:
            print(f"❌ Config loading failed: {e}")
            return False
        
        print("4. Testing standalone dashboard import...")
        try:
            from standalone_dashboard import StandaloneDashboard
            print("✅ StandaloneDashboard import successful")
        except Exception as e:
            print(f"❌ StandaloneDashboard import failed: {e}")
            print("Full traceback:")
            traceback.print_exc()
            return False
            
        print("5. Testing dashboard instance creation...")
        try:
            dashboard = StandaloneDashboard(port=dashboard_port, debug=False)
            print("✅ Dashboard instance created successfully")
        except Exception as e:
            print(f"❌ Dashboard instance creation failed: {e}")
            print("Full traceback:")
            traceback.print_exc()
            return False
            
        print("6. Testing what main_forecasting.py sees...")
        try:
            # Simulate what main_forecasting.py does
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Test the exact import sequence from main_forecasting.py
            from tensorflow_config import fix_tensorflow_configuration
            print("✅ TensorFlow config import works")
            
            # Test standalone dashboard import from main context
            from standalone_dashboard import StandaloneDashboard
            print("✅ StandaloneDashboard import works from main context")
            
        except Exception as e:
            print(f"❌ Main forecasting context failed: {e}")
            print("Full traceback:")
            traceback.print_exc()
            return False
            
        print("\n🎉 All imports and initialization tests passed!")
        print("The dashboard should work in main_forecasting.py")
        return True
        
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        traceback.print_exc()
        return False

def check_main_forecasting_issue():
    """Check what might be causing the main_forecasting.py issue"""
    print("\n🔍 Checking main_forecasting.py specific issues...")
    
    try:
        print("1. Checking if TensorFlow config interferes...")
        from tensorflow_config import fix_tensorflow_configuration
        fix_tensorflow_configuration()
        print("✅ TensorFlow config applied successfully")
        
        print("2. Checking parent directory imports...")
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)
        print(f"✅ Parent directory added: {parent_dir}")
        
        print("3. Testing mock modules...")
        import types
        mock_get_data = types.ModuleType('get_data')
        mock_get_data.get_processed_path = lambda: ("", "")
        sys.modules['get_data'] = mock_get_data
        print("✅ Mock modules created")
        
        print("4. Testing imports in order...")
        try:
            from initial_model import get_initial_model, get_online_data
            print("✅ Initial model imports work")
        except Exception as e:
            print(f"⚠️ Initial model import issues (may be expected): {e}")
            
        try:
            from online_forecasting_multi_step import multistep_rolling_buffer_learning_prediction_with_dash
            print("✅ Online forecasting import works")
        except Exception as e:
            print(f"⚠️ Online forecasting import issues (may be expected): {e}")
            
        print("5. Final dashboard import test...")
        from standalone_dashboard import StandaloneDashboard
        print("✅ Final dashboard import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Main forecasting context issue: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting dashboard debugging...")
    
    success1 = debug_dashboard_import()
    success2 = check_main_forecasting_issue()
    
    if success1 and success2:
        print("\n✅ Dashboard should work - the issue might be elsewhere!")
        print("   Check if main_forecasting.py is actually calling initialize_dashboard()")
    else:
        print("\n❌ Found issues that need to be resolved")
        print("   Dashboard initialization will fail until these are fixed")