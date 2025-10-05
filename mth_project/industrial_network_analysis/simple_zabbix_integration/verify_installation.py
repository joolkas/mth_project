#!/usr/bin/env python3
"""
🔍 Installation Verification Script

Checks if all required dependencies are properly installed.
"""

import sys
import subprocess

def check_package_installation():
    """Check if all required packages are installed"""
    print("🔍 Checking Package Installation")
    print("=" * 40)
    
    # Required packages and their import names
    packages = {
        'pyzabbix': 'pyzabbix',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'tensorflow': 'tensorflow',
        'plotly': 'plotly',
        'dash': 'dash',
        'flask': 'flask',
        'werkzeug': 'werkzeug',
        'urllib3': 'urllib3',
        'requests': 'requests'
    }
    
    optional_packages = {
        'keyboard': 'keyboard',
        'psutil': 'psutil',
        'schedule': 'schedule'
    }
    
    installed = []
    missing = []
    optional_missing = []
    
    print("📦 Checking required packages...")
    for pkg_name, import_name in packages.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {pkg_name}: {version}")
            installed.append(pkg_name)
        except ImportError:
            print(f"❌ {pkg_name}: NOT INSTALLED")
            missing.append(pkg_name)
    
    print("\n📦 Checking optional packages...")
    for pkg_name, import_name in optional_packages.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {pkg_name}: {version}")
        except ImportError:
            print(f"⚠️ {pkg_name}: not installed (optional)")
            optional_missing.append(pkg_name)
    
    print(f"\n📊 Summary:")
    print(f"✅ Installed: {len(installed)}/{len(packages)} required packages")
    print(f"❌ Missing: {len(missing)} required packages")
    print(f"⚠️ Optional missing: {len(optional_missing)} packages")
    
    if missing:
        print(f"\n🚨 Missing required packages: {', '.join(missing)}")
        print("To install missing packages:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n🎉 All required packages are installed!")
        return True

def test_dashboard_specific():
    """Test dashboard-specific functionality"""
    print("\n🎯 Testing Dashboard Functionality")
    print("=" * 40)
    
    try:
        print("1. Testing Dash import...")
        import dash
        print(f"✅ Dash {dash.__version__} imported successfully")
        
        print("2. Testing Plotly import...")
        import plotly
        print(f"✅ Plotly {plotly.__version__} imported successfully")
        
        print("3. Testing StandaloneDashboard import...")
        from standalone_dashboard import StandaloneDashboard
        print("✅ StandaloneDashboard imported successfully")
        
        print("4. Testing dashboard instance creation...")
        dashboard = StandaloneDashboard(port=8052, debug=False)
        print("✅ Dashboard instance created successfully")
        
        print("5. Testing configuration loading...")
        import json
        with open('config.json', 'r') as f:
            config = json.load(f)
        port = config['monitoring']['dashboard_port']
        print(f"✅ Config loaded, dashboard port: {port}")
        
        print("\n🎉 Dashboard functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        import traceback
        print("Full error:")
        traceback.print_exc()
        return False

def install_missing_packages():
    """Install missing packages using requirements.txt"""
    print("\n🔧 Installing Missing Packages")
    print("=" * 40)
    
    try:
        print("Running: pip install -r requirements.txt")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Package installation completed successfully")
            return True
        else:
            print(f"❌ Package installation failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Package installation timed out")
        return False
    except Exception as e:
        print(f"❌ Package installation error: {e}")
        return False

def main():
    """Main verification process"""
    print("🏭 Industrial Network Analysis - Installation Verification")
    print("=" * 60)
    
    # Step 1: Check current installation
    packages_ok = check_package_installation()
    
    if not packages_ok:
        print("\n❓ Would you like to install missing packages? (y/N): ", end='')
        try:
            response = input().strip().lower()
            if response in ['y', 'yes']:
                if install_missing_packages():
                    print("\n🔄 Re-checking package installation...")
                    packages_ok = check_package_installation()
                else:
                    print("❌ Package installation failed")
                    return False
            else:
                print("⚠️ Skipping package installation")
                return False
        except KeyboardInterrupt:
            print("\n🛑 Verification cancelled")
            return False
    
    # Step 2: Test dashboard functionality
    dashboard_ok = test_dashboard_specific()
    
    # Final result
    print("\n" + "=" * 60)
    if packages_ok and dashboard_ok:
        print("🎉 VERIFICATION PASSED")
        print("✅ All dependencies are installed correctly")
        print("✅ Dashboard functionality works")
        print("\n🚀 Your system should now work without 'dash_plotter is None' errors!")
        print("   Run: python main_forecasting.py")
        return True
    else:
        print("❌ VERIFICATION FAILED")
        print("   Please resolve the issues above before running the main system")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)