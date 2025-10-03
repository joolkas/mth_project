# 🏭 Simplified Zabbix Industrial Anomaly Detection

**Clean, minimal integration that connects your trained anomaly detection models to live industrial data from Zabbix.**

## ✨ What This System Does

- **Fetches real-time data** from industrial devices via Zabbix API
- **Runs your existing LSTM models** without any modifications
- **Displays anomaly detection results** on your Dash dashboard
- **READ-ONLY operation** - no data is sent back to Zabbix
- **1-minute monitoring cycle** - same as your original system

## 🚀 Quick Setup

### 1. Deploy the System
```bash
# Make deployment script executable
chmod +x deploy.sh

# Deploy (as root for system-wide installation)
sudo ./deploy.sh

# Or deploy to current directory
./deploy.sh
```

### 2. Configure Zabbix Connection
Edit `config.json`:
```json
{
  "zabbix": {
    "url": "http://your-zabbix-server/zabbix",
    "user": "Admin",
    "password": "your-password"
  },
  "models": {
    "forecasting_model_path": "../forecasting_model"
  },
  "monitoring": {
    "host_groups": ["Industrial", "SCADA", "Network", "PLC"],
    "update_interval": 60
  }
}
```

### 3. Test the System
```bash
# Test Zabbix connection and model loading
./test.sh

# Or if installed system-wide
sudo python3 /opt/anomaly_detection/zabbix_monitor.py --test

# For detailed connection testing
sudo python3 /opt/anomaly_detection/test_zabbix.py
```

**💡 If connection fails**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed debugging steps.

### 4. Start Monitoring
```bash
# If installed as service
sudo systemctl start zabbix-anomaly-monitor

# Or run directly
./start.sh
```

### 5. View Results
- **Dashboard**: http://localhost:8050
- **Logs**: `sudo journalctl -u zabbix-anomaly-monitor -f`

## 📊 How It Works

### Data Flow
1. **Zabbix API** → Fetch metrics from industrial devices  
2. **Data Preparation** → Format data to match your model's requirements
3. **LSTM Forecasting** → Run your existing `multistep_rolling_buffer_learning_prediction_with_dash`
4. **Anomaly Detection** → Compare predictions vs actual values
5. **Dashboard** → Display results in real-time

### Integration Points
- Uses your **existing trained models** (no retraining needed)
- Preserves your **1-minute monitoring cycle**
- Maintains your **Dash dashboard interface**
- Compatible with your **classification models**

## 🔧 System Requirements

### Zabbix Setup
1. **Host Groups**: Organize devices into groups like "Industrial", "SCADA", "PLC"
2. **Monitored Items**: Ensure devices collect relevant metrics:
   - CPU utilization
   - Memory usage  
   - Network traffic
   - Temperature sensors
   - Custom industrial metrics

### Model Requirements
- Your trained LSTM model in `forecasting_model/` directory
- Saved scalers and preprocessing parameters
- Variable definitions matching your training data

## 🏭 Files Structure

```
zabbix_integration_clean/
├── zabbix_monitor.py    # Main monitoring script (400 lines)
├── config.json          # Simple configuration
├── requirements.txt     # Minimal dependencies
├── deploy.sh           # One-click deployment
└── README.md           # This file
```

**Removed Complexity:**
- ❌ Duplicated core modules (`initial_model.py`, `data_utils.py`, etc.)
- ❌ Multiple complex scripts (`main.py`, `simple_main.py`, `auto_train.py`, etc.)
- ❌ Multiple deployment options
- ❌ Complex configuration files
- ❌ Unnecessary training scripts

## 🎯 Key Features

### ✅ What's Included
- **Minimal codebase** - single 400-line script
- **Automatic data adaptation** - handles varying Zabbix metrics  
- **Error resilience** - continues monitoring despite individual failures
- **Simple configuration** - just 4 key settings
- **Easy deployment** - single script setup

### ✅ What's Preserved
- **Your original anomaly detection algorithm**
- **Your trained LSTM models**
- **Your Dash dashboard interface**  
- **Your 1-minute monitoring cycle**
- **Your classification capabilities**

## 🚀 Production Ready

This simplified system is ready for deployment in industrial environments:

- **Minimal dependencies** - only essential packages
- **Clean codebase** - easy to maintain and modify
- **Robust error handling** - handles network issues gracefully
- **System service** - runs automatically on boot
- **Comprehensive logging** - easy troubleshooting

Transform your research into a **production industrial monitoring solution** with live device data! 🏭✨