# 🏭 Simple Zabbix Industrial Monitoring

**Ultra-simplified version that automatically adapts to any Zabbix environment.**

## ✨ What It Does

- **🔍 Auto-Discovery**: Finds all hosts and metrics automatically
- **🔧 Auto-Adaptation**: Adapts to any number of features (23, 42, 100+)
- **🧠 Smart Selection**: Picks the most important features automatically
- **💾 Auto-Save**: Saves current variables for reference
- **🔄 Auto-Recovery**: Handles errors and continues running

## 🚀 Quick Start

### 1. Deploy
```bash
cd zabbix_integration
sudo ./simple_deploy.sh
```

### 2. Test Connection
```bash
python3 /opt/anomaly_detection/main.py --test
```

### 3. Collect Training Data
```bash
python3 /opt/anomaly_detection/simple_training.py
```

### 4. Train Model
Use the generated `training_data/forecasting.csv` with your existing training pipeline.

### 5. Start Monitoring
```bash
sudo systemctl start zabbix-monitoring
sudo journalctl -u zabbix-monitoring -f
```

## ⚙️ Configuration

Only need to configure `config.json`:

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
  "industrial_filters": {
    "device_groups": ["Zabbix servers", "Virtual machines", "Industrial"]
  }
}
```

## 🎯 Key Features

### **Complete Auto-Adaptation**
- Changes `device_groups` → System adapts automatically
- Different hosts → Discovers and uses new metrics
- More/fewer features → Auto-selects optimal subset
- New environment → Works without code changes

### **Intelligent Feature Selection** 
Automatically prioritizes:
- 🧠 Memory metrics (available, used, total)
- 🖥️ CPU metrics (utilization, processes)
- 🌐 Network metrics (bits, packets, interfaces)
- 💾 Disk metrics (space available, used)
- 🌡️ Sensor metrics (temperature, pressure)

### **Zero Configuration**
- No manual feature mapping
- No hardcoded variable names
- No feature count limits
- No device-specific code

## 📊 How It Works

1. **Discovery**: Scans configured host groups
2. **Collection**: Gets all available metrics
3. **Adaptation**: Automatically selects best features for model
4. **Processing**: Applies same preprocessing as training
5. **Prediction**: Runs ML model with adapted data
6. **Logging**: Records everything for monitoring

## 🔧 Troubleshooting

### Connection Issues
```bash
# Check Zabbix connection
python3 /opt/anomaly_detection/main.py --test
```

### No Data
- Verify host groups exist in Zabbix
- Check hosts have monitored items
- Ensure Zabbix user has permissions

### Feature Mismatch
- System automatically adapts - no action needed
- Retrain model with current data for best results

### Service Issues
```bash
# Check service status
sudo systemctl status zabbix-monitoring

# View logs
sudo journalctl -u zabbix-monitoring -f

# Restart service
sudo systemctl restart zabbix-monitoring
```

## 📈 Example Output

```
🎯 Starting monitoring (expecting 23 features)
🔄 Cycle #1
📊 Collected data: (120, 42)
🔧 Auto-selecting 23 from 42 features
💾 Saved 23 variables to variables.txt
✅ Cycle #1 completed successfully
```

## 🚀 Advanced Usage

### Custom Training Data Collection
```bash
# Collect specific time period
python3 -c "
from simple_training import collect_training_data
collect_training_data(days=14)  # 14 days of data
"
```

### Manual Feature Count
Modify `simple_main.py` line 185:
```python
expected_features = 30  # Force specific count
```

This simplified system handles **everything automatically** - just change the config and it adapts! 🎯