# 🚀 Service Troubleshooting Guide

## 🎯 Your Situation
- ✅ **Test passed** - Zabbix connection and models work
- ❌ **Service fails** - No logs, no dashboard
- ❌ **Silent failure** - Service appears to start but does nothing

## 🔍 Step-by-Step Diagnosis

### **1. Run the Diagnostic Script**
```bash
cd /opt/anomaly_detection
chmod +x diagnose.sh
./diagnose.sh
```

This will show:
- Service status and logs
- Running processes
- Port usage (Dash server)
- Configuration validation

### **2. Try Manual Execution**
```bash
chmod +x run_manual.sh
./run_manual.sh
```

This runs the monitor in **foreground mode** so you can see all output directly.

### **3. Use Debug Version**
```bash
chmod +x debug_monitor.py
python3 debug_monitor.py
```

This has extensive logging and shows exactly where failures occur.

## 🔧 Common Issues & Solutions

### **Issue 1: Service Starts But Does Nothing**
**Symptom**: `systemctl status` shows "active" but no logs in `journalctl`

**Cause**: Python import failures or path issues

**Solution**:
```bash
# Stop service
sudo systemctl stop zabbix-anomaly-monitor

# Try manual run to see errors
cd /opt/anomaly_detection
python3 debug_monitor.py
```

### **Issue 2: Dash Server Won't Start**
**Symptom**: No port 8050 listening, dashboard inaccessible

**Causes**:
- Port already in use
- Permissions issues  
- Import failures

**Solutions**:
```bash
# Check if port is in use
netstat -tulpn | grep :8050

# Kill any existing Dash processes
pkill -f "dash"

# Try starting manually
cd /opt/anomaly_detection
python3 -c "
try:
    import sys
    sys.path.append('/opt/anomaly_detection/../industrial_network_analysis')
    from dash_plotter import DashRealTimePlotter
    plotter = DashRealTimePlotter()
    plotter.start_server()
    print('✅ Dash server started successfully')
    import time
    time.sleep(5)
except Exception as e:
    print(f'❌ Dash server failed: {e}')
    import traceback
    traceback.print_exc()
"
```

### **Issue 3: Import Path Problems**
**Symptom**: "ModuleNotFoundError" for your original modules

**Solution**: Fix Python path in service
```bash
# Edit the service file
sudo systemctl edit --full zabbix-anomaly-monitor

# Add this line in [Service] section:
Environment=PYTHONPATH=/opt/anomaly_detection/../industrial_network_analysis

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart zabbix-anomaly-monitor
```

### **Issue 4: Permission Issues**
**Symptom**: Service fails to start, permission denied errors

**Solutions**:
```bash
# Fix ownership
sudo chown -R root:root /opt/anomaly_detection
sudo chmod +x /opt/anomaly_detection/zabbix_monitor.py

# Check model directory permissions
sudo chmod -R 755 /opt/anomaly_detection/models
```

## 🎯 Quick Service Reset

If all else fails, try a complete reset:

```bash
# Stop and disable service
sudo systemctl stop zabbix-anomaly-monitor
sudo systemctl disable zabbix-anomaly-monitor

# Remove service file
sudo rm /etc/systemd/system/zabbix-anomaly-monitor.service

# Redeploy
cd /path/to/zabbix_integration_clean
sudo ./deploy.sh

# Test manually first
cd /opt/anomaly_detection
./run_manual.sh
```

## 🚀 Expected Working Output

When everything works correctly, you should see:

### **In Service Logs** (`journalctl -u zabbix-anomaly-monitor -f`):
```
Oct 03 22:45:01 zabbix-python systemd[1]: Started Zabbix Industrial Anomaly Detection Monitor.
Oct 03 22:45:02 zabbix-python python3[12345]: 🏭 Starting industrial anomaly monitoring...
Oct 03 22:45:03 zabbix-python python3[12345]: ✅ Connected to Zabbix 6.4.0
Oct 03 22:45:04 zabbix-python python3[12345]: ✅ Model loaded successfully
Oct 03 22:45:05 zabbix-python python3[12345]: ✅ Dash server started at http://localhost:8050
Oct 03 22:45:06 zabbix-python python3[12345]: 🔄 Monitoring cycle #1
Oct 03 22:45:07 zabbix-python python3[12345]: 📊 Collected data: (120, 15)
Oct 03 22:45:08 zabbix-python python3[12345]: ✅ Cycle #1 completed successfully
```

### **Dashboard Access**:
- URL: http://192.168.93.45:8050 (or http://localhost:8050)
- Should show real-time charts updating every minute

## 💡 Most Likely Issue

Based on your symptoms, the most likely cause is **Python import path issues** when running as a service. The service can't find your original anomaly detection modules.

**Quick Fix**: Run the manual version first to see exact error messages, then fix the service configuration based on what you find.

Run `./diagnose.sh` and `./run_manual.sh` - they'll show you exactly what's going wrong! 🔍✨