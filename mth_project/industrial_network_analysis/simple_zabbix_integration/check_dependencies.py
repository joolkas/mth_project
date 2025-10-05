#!/usr/bin/env python3
"""
Dependency Checker for Industrial Network Anomaly Detection System
Verifies all required packages are installed and working correctly.
"""

import sys
import subprocess
import importlib

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 PYTHON VERSION CHECK")
    print("=" * 40)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False

def check_package(package_name, import_name=None, version_attr=None):
    """Check if a package is installed and working"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        
        # Try to get version if specified
        if version_attr and hasattr(module, version_attr):
            version = getattr(module, version_attr)
            print(f"✅ {package_name}: {version}")
        else:
            print(f"✅ {package_name}: Available")
        return True
        
    except ImportError as e:
        print(f"❌ {package_name}: Not installed ({e})")
        return False
    except Exception as e:
        print(f"⚠️ {package_name}: Import error ({e})")
        return False

def check_core_dependencies():
    """Check core dependencies for the monitoring system"""
    print("\n📦 CORE DEPENDENCIES CHECK")
    print("=" * 40)
    
    dependencies = [
        ("pandas", "pandas", "__version__"),
        ("numpy", "numpy", "__version__"),
        ("pyzabbix", "pyzabbix", None),
        ("urllib3", "urllib3", "__version__"),
        ("requests", "requests", "__version__"),
    ]
    
    results = []
    for dep in dependencies:
        result = check_package(*dep)
        results.append(result)
    
    return all(results)

def check_dashboard_dependencies():
    """Check dashboard dependencies"""
    print("\n📊 DASHBOARD DEPENDENCIES CHECK")
    print("=" * 40)
    
    dashboard_deps = [
        ("dash", "dash", "__version__"),
        ("plotly", "plotly", "__version__"),
    ]
    
    results = []
    for dep in dashboard_deps:
        result = check_package(*dep)
        results.append(result)
    
    return all(results)

def check_optional_dependencies():
    """Check optional dependencies"""
    print("\n🔧 OPTIONAL DEPENDENCIES CHECK")
    print("=" * 40)
    
    optional_deps = [
        ("scikit-learn", "sklearn", "__version__"),
        ("scipy", "scipy", "__version__"),
        ("psutil", "psutil", "__version__"),
        ("statsmodels", "statsmodels", "__version__"),
    ]
    
    results = []
    for dep in optional_deps:
        result = check_package(*dep)
        results.append(result)
    
    return results

def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\n🧪 FUNCTIONALITY TESTS")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 0
    
    # Test pandas
    total_tests += 1
    try:
        import pandas as pd
        df = pd.DataFrame({'test': [1, 2, 3]})
        assert len(df) == 3
        print("✅ Pandas: DataFrame operations working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Pandas: Test failed ({e})")
    
    # Test numpy
    total_tests += 1
    try:
        import numpy as np
        arr = np.array([1, 2, 3])
        assert np.mean(arr) == 2.0
        print("✅ NumPy: Array operations working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ NumPy: Test failed ({e})")
    
    # Test dashboard components
    total_tests += 1
    try:
        import dash
        from dash import dcc, html
        import plotly.graph_objs as go
        
        # Create a simple plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 2]))
        print("✅ Dashboard: Plot creation working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Dashboard: Test failed ({e})")
    
    # Test pyzabbix
    total_tests += 1
    try:
        from pyzabbix import ZabbixAPI
        # Just test import, don't try to connect
        print("✅ PyZabbix: Import successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ PyZabbix: Test failed ({e})")
    
    return tests_passed, total_tests

def generate_install_commands():
    """Generate installation commands for missing packages"""
    print("\n💿 INSTALLATION COMMANDS")
    print("=" * 40)
    
    print("If any packages are missing, install them with:")
    print("")
    print("# Core dependencies:")
    print("pip install pandas numpy pyzabbix urllib3 requests")
    print("")
    print("# Dashboard dependencies:")
    print("pip install dash plotly")
    print("")
    print("# Optional dependencies:")
    print("pip install scikit-learn scipy psutil statsmodels")
    print("")
    print("# All at once:")
    print("pip install pandas numpy pyzabbix urllib3 requests dash plotly scikit-learn scipy psutil statsmodels")

def main():
    """Main dependency check routine"""
    print("🔍 INDUSTRIAL NETWORK ANOMALY DETECTION")
    print("   DEPENDENCY CHECKER")
    print("=" * 50)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check dependencies
    core_ok = check_core_dependencies()
    dashboard_ok = check_dashboard_dependencies()
    optional_results = check_optional_dependencies()
    
    # Run functionality tests
    tests_passed, total_tests = test_basic_functionality()
    
    # Summary
    print("\n📋 SUMMARY")
    print("=" * 40)
    
    if python_ok:
        print("✅ Python version: OK")
    else:
        print("❌ Python version: UPGRADE NEEDED")
    
    if core_ok:
        print("✅ Core dependencies: OK")
    else:
        print("❌ Core dependencies: MISSING")
    
    if dashboard_ok:
        print("✅ Dashboard dependencies: OK")
    else:
        print("❌ Dashboard dependencies: MISSING")
    
    optional_count = sum(optional_results)
    print(f"📊 Optional dependencies: {optional_count}/4 available")
    
    print(f"🧪 Functionality tests: {tests_passed}/{total_tests} passed")
    
    # Overall status
    print("\n🎯 OVERALL STATUS")
    print("=" * 40)
    
    if python_ok and core_ok and tests_passed >= 3:
        print("🎉 SYSTEM READY! Your environment is properly configured.")
        print("   You can run the industrial network monitoring system.")
        
        if dashboard_ok:
            print("📊 Dashboard will be available at: http://localhost:8052")
        else:
            print("⚠️  Dashboard not available (install dash and plotly)")
            
    elif python_ok and core_ok:
        print("⚠️  MOSTLY READY: Core functionality available but some tests failed.")
        print("   Basic monitoring should work, but check error messages above.")
    else:
        print("❌ NOT READY: Critical dependencies missing.")
        print("   Install missing packages before running the monitoring system.")
    
    # Generate install commands if needed
    if not core_ok or not dashboard_ok:
        generate_install_commands()
    
    return python_ok and core_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)