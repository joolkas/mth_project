#!/usr/bin/env python3
"""
🧪 Simple Test Script for Zabbix Integration

Tests the basic functionality without requiring parent directory dependencies.
"""

import sys
import os
import json

def test_basic_imports():
    """Test basic Python package imports"""
    print("🐍 Testing basic imports...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        from pyzabbix import ZabbixAPI
        print("✅ PyZabbix imported successfully")
    except ImportError as e:
        print(f"❌ PyZabbix import failed: {e}")
        return False
    
    try:
        import plotly
        import dash
        print("✅ Plotly and Dash imported successfully")
    except ImportError as e:
        print(f"❌ Plotly/Dash import failed: {e}")
        return False
    
    return True

def test_config_file():
    """Test config file loading"""
    print("\n📋 Testing configuration...")
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        required_keys = ['zabbix', 'data_collection', 'model', 'monitoring']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing config key: {key}")
                return False
        
        print("✅ Configuration file loaded successfully")
        return True
    except FileNotFoundError:
        print("❌ config.json not found")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config.json: {e}")
        return False

def test_model_functions():
    """Test model creation functions"""
    print("\n🤖 Testing model functions...")
    
    try:
        # Test if our train_model.py can be imported
        sys.path.insert(0, os.getcwd())
        
        # Create a simple test to see if TensorFlow model can be created
        import tensorflow as tf
        
        # Simple model test
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(60, 10)),
            tf.keras.layers.Dense(10)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        print("✅ TensorFlow LSTM model creation works")
        return True
        
    except Exception as e:
        print(f"❌ Model function test failed: {e}")
        return False

def test_data_collector():
    """Test data collector initialization"""
    print("\n📊 Testing data collector...")
    
    try:
        from get_data import ZabbixDataCollector
        
        collector = ZabbixDataCollector('config.json')
        print("✅ ZabbixDataCollector initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Data collector test failed: {e}")
        return False

def test_directories():
    """Test directory structure"""
    print("\n📁 Testing directories...")
    
    directories = ['data', 'temp_data', 'trained_model']
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"✅ Created directory: {directory}")
            except Exception as e:
                print(f"❌ Failed to create directory {directory}: {e}")
                return False
        else:
            print(f"✅ Directory exists: {directory}")
    
    return True

def main():
    """Run all tests"""
    print("🧪 Zabbix Integration - Standalone System Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_config_file),
        ("Model Functions", test_model_functions),
        ("Data Collector", test_data_collector),
        ("Directories", test_directories)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"💥 {test_name} test failed")
        except Exception as e:
            print(f"💥 {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        print("\n🚀 Next steps:")
        print("1. Configure config.json with your Zabbix details")
        print("2. Test Zabbix connection: python get_data.py --test")
        print("3. Collect training data: python get_data.py --collect-history --hours 48")
        print("4. Train model: python train_model.py --data data/historical_data_*.csv")
        print("5. Start monitoring: python main_forecasting.py")
    else:
        print(f"⚠️  {total - passed} tests failed. Please resolve issues before proceeding.")
        print("\n💡 Common solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Check config.json format and required keys")
        print("- Ensure you have write permissions in current directory")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)