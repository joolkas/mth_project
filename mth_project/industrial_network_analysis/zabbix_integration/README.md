# 🏭 Simple Zabbix Industrial Monitoring

**READ-ONLY monitoring system that fetches data from Zabbix and runs your trained ML models for industrial anomaly detection.**

## ✨ What It Does

- **Fetches data FROM Zabbix** (no data sent back)
- **Runs your existing LSTM forecasting model** 
- **Uses your 1-minute monitoring cycle**
- **Replaces `df_online` with real industrial device data**
- **Shows results on Dash dashboard**

## 🚀 Quick Setup

### 1. Install on Debian VM
```bash
# Clone repository
git clone -b official_program https://github.com/joolkas/mth_project.git
cd mth_project/mth_project/industrial_network_analysis/zabbix_integration

# Deploy
chmod +x scripts/deploy.sh
sudo ./scripts/deploy.sh
```

### 2. Copy Your Models
```bash
# Copy your trained forecasting model
sudo cp -r ../forecasting_model /opt/anomaly_detection/models/
```

### 3. Start Monitoring
```bash
# Start service
sudo systemctl start zabbix-monitoring

# View logs
sudo journalctl -u zabbix-monitoring -f

# Open dashboard: http://localhost:8050
```

## ⚙️ Configuration

Simple config in `config.json`:

```json
{
  "zabbix": {
    "url": "http://your-zabbix-server/zabbix",
    "user": "Admin",
    "password": "your-password"
  },
  "models": {
    "forecasting_model_path": "/opt/anomaly_detection/models/forecasting_model"
  },
  "monitoring": {
    "update_interval": 60
  },
  "industrial_filters": {
    "device_groups": ["Industrial", "SCADA", "PLC", "HMI", "Network"]
  }
}
```

## 🔧 Usage

### Test Connection
```bash
python3 /opt/anomaly_detection/main.py --test
```

### Run Monitoring
```bash
python3 /opt/anomaly_detection/main.py
```

### Service Management
```bash
# Start/stop service
sudo systemctl start zabbix-monitoring
sudo systemctl stop zabbix-monitoring

# Check status
sudo systemctl status zabbix-monitoring

# View logs
sudo journalctl -u zabbix-monitoring -f
```

## 📊 Data Mapping

Your training variables automatically map to Zabbix metrics:

| Training Variable | Zabbix Items |
|-------------------|--------------|
| ICMP response time | `icmpping`, `icmppingsec` |
| Switch Temperature | `sensor.temp.*`, `temperature.*` |
| CPU utilization | `system.cpu.util` |
| Memory utilization | `vm.memory.util` |
| Network traffic | `net.if.in`, `net.if.out` |

## 🏭 Zabbix Requirements

1. **Host Groups**: Create groups named "Industrial", "SCADA", "PLC", "HMI", "Network"
2. **Add Devices**: Assign your industrial devices to these groups  
3. **Enable Metrics**: Ensure devices collect required items (CPU, memory, network, sensors)

## 🎯 Integration

The system seamlessly integrates with your existing code:

- **Your `online_forecasting_multi_step.py`**: Works unchanged
- **Your trained models**: Used as-is
- **Your Dash dashboard**: Shows real industrial data
- **1-minute cycle**: Maintained exactly as designed

This transforms your research system into a **production industrial monitoring solution** with live device data! 🏭✨