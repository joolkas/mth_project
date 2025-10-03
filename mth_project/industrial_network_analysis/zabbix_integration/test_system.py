#!/usr/bin/env python3
"""
Simple test script to verify the Zabbix integration system
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from main import OptimizedZabbixConnector
        print("✅ Zabbix connector import successful")
    except Exception as e:
        print(f"❌ Zabbix connector import failed: {e}")
        return False
    
    try:
        from initial_model import get_initial_model
        print("✅ Initial model import successful")
    except Exception as e:
        print(f"❌ Initial model import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} import successful")
    except Exception as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")
    
    try:
        connector = OptimizedZabbixConnector("config.json")
        print("✅ Configuration loaded successfully")
        print(f"   Update interval: {connector.update_interval}s")
        print(f"   Context length: {connector.context_length}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    print("\n🧪 Testing model loading...")
    
    try:
        from initial_model import get_initial_model
        
        # Try loading with default path first
        model_path = "../forecasting_model"
        if not os.path.exists(model_path):
            print(f"⚠️  Model path {model_path} not found, trying absolute path...")
            model_path = "/opt/anomaly_detection/models"
        
        if os.path.exists(model_path):
            model = get_initial_model(model_path)
            print("✅ Model loaded successfully")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            return True
        else:
            print(f"❌ Model path {model_path} not found")
            return False
            
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting system verification tests...")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Model Loading", test_model_loading)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed! System is ready.")
        return True
    else:
        print("❌ Some tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)