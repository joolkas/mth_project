# 🏭 Service-Based Deployment Guide

## 🚀 **Production Service Installation**

### **Option 1: RHEL/CentOS Production Service**
```bash
# 1. Install as systemd service
sudo ./rhel_install.sh

# 2. Configure Zabbix connection  
sudo nano /opt/zabbix_anomaly_detection/config.json

# 3. Test connection
/opt/zabbix_anomaly_detection/test.sh

# 4. Collect training data
sudo -u anomaly python3 /opt/zabbix_anomaly_detection/get_data.py --collect-history --hours 48

# 5. Train model
sudo -u anomaly python3 /opt/zabbix_anomaly_detection/train_model.py --data /opt/zabbix_anomaly_detection/data/historical_data_*.csv

# 6. Start service
/opt/zabbix_anomaly_detection/start.sh

# 7. Check status
/opt/zabbix_anomaly_detection/status.sh
```

### **Service Management Commands:**
```bash
# Start the service
sudo systemctl start zabbix-anomaly-detection

# Stop the service  
sudo systemctl stop zabbix-anomaly-detection

# Check status
sudo systemctl status zabbix-anomaly-detection

# Enable auto-start on boot
sudo systemctl enable zabbix-anomaly-detection

# View logs
sudo journalctl -u zabbix-anomaly-detection -f
```

---

## 🎯 **Service vs Direct Execution**

### **Direct Execution (Development/Testing):**
- ✅ Use for development and testing
- ✅ Run directly: `python main_forecasting.py`
- ✅ Easy debugging and modification
- ✅ Manual start/stop control

### **Service Mode (Production):**
- ✅ Auto-starts on system boot
- ✅ Runs as dedicated service user
- ✅ Auto-restarts on failures
- ✅ Integrated logging with systemd
- ✅ SELinux security policies
- ✅ Resource limits and monitoring
- ✅ Log rotation configured

---

## 🔧 **Service Features**

### **Auto-Restart on Failure:**
```ini
Restart=always
RestartSec=10
StartLimitInterval=60
StartLimitBurst=3
```

### **Security Hardening:**
- Dedicated `anomaly` user account  
- SELinux policies configured
- Resource limits applied
- Network firewall rules

### **Monitoring Integration:**
- SystemD journal logging
- Log rotation configured
- Status monitoring scripts
- Health check endpoints

---

## 📋 **Choose Your Deployment Method**

### **For Development/Testing:**
Follow the original step-by-step guide with direct Python execution

### **For Production:**
Use the service installation:
```bash
sudo ./rhel_install.sh
# Follow the service configuration steps above
```

Both methods run the **exact same code** - just different deployment approaches! 🎉