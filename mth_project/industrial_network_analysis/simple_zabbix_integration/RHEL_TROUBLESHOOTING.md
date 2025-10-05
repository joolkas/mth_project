# 🔧 RHEL Installation Troubleshooting Guide

## The Error You Encountered

The error you're seeing is a common Zabbix package dependency conflict on RHEL 9. Here's what's happening:

```
Problem 1: package zabbix-server-pgsql-1:6.0.41-1.el9.x86_64 requires zabbix = 1:6.0.41-1.el9
```

**Root Cause:** Mixed Zabbix versions (6.0 and 7.0) and SELinux policy version conflicts.

## 🚀 Quick Fix (Recommended)

### Option 1: Use the Fixed Installation Script
```bash
# Make the script executable
chmod +x rhel_install_fixed.sh

# Run the fixed installation
./rhel_install_fixed.sh
```

This script:
- ✅ Resolves Zabbix version conflicts
- ✅ Sets up Python environment properly  
- ✅ Creates project structure
- ✅ Installs all dependencies

### Option 2: Clean Zabbix Installation
```bash
# Make cleanup script executable
chmod +x zabbix_cleanup.sh

# Run as root to clean up Zabbix conflicts
sudo ./zabbix_cleanup.sh
```

## 🐍 Python Environment Setup (Independent of Zabbix)

If you just want to get the Python monitoring working without fixing Zabbix:

```bash
# Install Python 3.11 and development tools
sudo dnf install -y python3.11 python3.11-devel python3.11-pip
sudo dnf install -y gcc gcc-c++ make openssl-devel libffi-devel

# Create virtual environment
python3.11 -m venv monitoring_env
source monitoring_env/bin/activate

# Install monitoring dependencies
pip install pandas numpy pyzabbix dash plotly urllib3 requests

# Check if everything works
python check_dependencies.py
```

## 🔍 Dependency Check

Always run this after installation:
```bash
python check_dependencies.py
```

This will verify:
- ✅ Python version compatibility
- ✅ Core monitoring packages
- ✅ Dashboard dependencies
- ✅ Basic functionality tests

## 🏭 Using Your Monitoring System

Once dependencies are installed, you can use your simplified monitoring system:

```bash
# Test the fixed version
python main_forecasting_fixed.py --test

# Run monitoring (with working dashboard)
python main_forecasting_fixed.py
```

## 🛠️ Manual Dependency Installation

If the automated scripts don't work, install manually:

```bash
# Core system packages
sudo dnf update -y
sudo dnf install -y epel-release
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y python3.11 python3.11-devel python3.11-pip
sudo dnf install -y gcc gcc-c++ make openssl-devel libffi-devel

# Python packages
pip install --user pandas>=1.5.0 numpy>=1.21.0 pyzabbix>=1.0.0
pip install --user dash>=2.10.0 plotly>=5.15.0
pip install --user urllib3>=1.26.0 requests>=2.28.0
pip install --user psutil>=5.9.0 scikit-learn>=1.1.0
```

## 🚨 Common Issues and Solutions

### Issue 1: "Package conflicts"
**Solution:** Use the cleanup script to remove all Zabbix packages first
```bash
sudo ./zabbix_cleanup.sh
```

### Issue 2: "Python module not found"
**Solution:** Ensure virtual environment is activated
```bash
source venv/bin/activate  # or monitoring_env/bin/activate
python check_dependencies.py
```

### Issue 3: "Permission denied"
**Solution:** Don't run Python scripts as root
```bash
# Wrong:
sudo python main_forecasting_fixed.py

# Correct:
python main_forecasting_fixed.py
```

### Issue 4: "Dashboard not loading"
**Solution:** Check if dashboard dependencies are installed
```bash
pip install dash plotly
```

### Issue 5: "Zabbix connection failed"
**Solution:** Your monitoring system can work without Zabbix server installed
- Use `main_forecasting_fixed.py` for basic monitoring
- Configure it to connect to remote Zabbix server
- Or run in simulation mode for testing

## 📋 Verification Checklist

After installation, verify these work:

- [ ] `python3.11 --version` shows Python 3.11+
- [ ] `python check_dependencies.py` shows all green checkmarks
- [ ] `python main_forecasting_fixed.py --test` connects (or shows clear error)
- [ ] Dashboard loads at `http://localhost:8052` (if dash/plotly installed)

## 🎯 What Each File Does

| File | Purpose |
|------|---------|
| `rhel_install_fixed.sh` | Complete installation with Zabbix conflict resolution |
| `zabbix_cleanup.sh` | Removes conflicting Zabbix packages |
| `check_dependencies.py` | Verifies Python environment |
| `main_forecasting_fixed.py` | Your simplified monitoring system |
| `compare_systems.py` | Shows differences between original and fixed |

## 🚀 Success Path

1. **Clean slate:** `sudo ./zabbix_cleanup.sh`
2. **Install properly:** `./rhel_install_fixed.sh`
3. **Verify environment:** `python check_dependencies.py`
4. **Test monitoring:** `python main_forecasting_fixed.py --test`
5. **Run monitoring:** `python main_forecasting_fixed.py`

## 📞 Still Having Issues?

If you're still having problems:

1. Check the logs: `tail -f logs/monitor.log`
2. Run dependency check: `python check_dependencies.py`
3. Verify Python environment: `which python` and `python --version`
4. Test individual components: `python -c "import pandas; print('OK')"`

The simplified system (`main_forecasting_fixed.py`) is designed to work with minimal dependencies and should run even if Zabbix server installation fails.