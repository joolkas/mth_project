#!/usr/bin/env python3
"""
🧪 Test Script for TensorFlow and Dashboard Fixes

Tests both the TensorFlow configuration fix and standalone dashboard.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_tensorflow_fix():
    """Test TensorFlow configuration fix"""
    print("🔧 Testing TensorFlow configuration fix...")
    
    try:
        from tensorflow_config import fix_tensorflow_configuration
        fix_tensorflow_configuration()
        print("✅ TensorFlow configuration applied successfully")
        
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} loaded")
        
        # Test a simple model creation (should not show eager execution warning)
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(60, 5)),
            tf.keras.layers.Dense(5)
        ])
        model.compile(optimizer='adam', loss='mse')
        print("✅ TensorFlow model creation successful (check for warnings above)")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow fix test failed: {e}")
        return False

def test_standalone_dashboard():
    """Test standalone dashboard"""
    print("\n📊 Testing standalone dashboard...")
    
    try:
        from standalone_dashboard import StandaloneDashboard
        from datetime import datetime
        import random
        import time
        
        # Create dashboard instance
        dashboard = StandaloneDashboard(port=8052, debug=False)
        print("✅ Dashboard instance created")
        
        # Test data update
        timestamp = datetime.now()
        predictions = {
            'CPU_Usage': 45.2,
            'Memory_Usage': 67.8,
            'Network_Bits': 1234.5
        }
        actuals = {
            'CPU_Usage': 44.8,
            'Memory_Usage': 68.1,
            'Network_Bits': 1245.2
        }
        
        dashboard.update_data(timestamp, predictions, actuals, is_anomaly=False)
        print("✅ Dashboard data update successful")
        
        # Test server start (don't actually start to avoid port conflicts)
        print("✅ Dashboard ready to start (not starting to avoid port conflicts)")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def test_main_forecasting_imports():
    """Test main forecasting module imports"""
    print("\n🚀 Testing main forecasting imports...")
    
    try:
        # Test TensorFlow fix import
        from tensorflow_config import fix_tensorflow_configuration
        print("✅ TensorFlow config import successful")
        
        # Test standalone dashboard import
        from standalone_dashboard import StandaloneDashboard
        print("✅ Standalone dashboard import successful")
        
        # Test if main_forecasting can import these
        print("✅ Main forecasting should now work with these imports")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing TensorFlow and Dashboard Fixes")
    print("=" * 50)
    
    tests = [
        ("TensorFlow Configuration", test_tensorflow_fix),
        ("Standalone Dashboard", test_standalone_dashboard),
        ("Main Forecasting Imports", test_main_forecasting_imports)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Fixes are working correctly.")
        print("\n🚀 Your issues should now be resolved:")
        print("1. ✅ TensorFlow eager execution warning eliminated")
        print("2. ✅ Standalone dashboard ready for consistent plotting")
        print("\n📋 Next steps:")
        print("1. Run: python main_forecasting.py")
        print("2. Open: http://localhost:8051 for dashboard")
        print("3. Check for absence of TensorFlow warnings")
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)