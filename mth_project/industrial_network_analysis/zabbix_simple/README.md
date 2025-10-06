# 🏭 Simple Zabbix Integration for Industrial Network Anomaly Detection

A streamlined, production-ready integration that connects your existing LSTM forecasting models to live industrial data from Zabbix systems on Debian/Ubuntu servers.

## ✨ What This System Does

This simplified integration provides:

1. **🔍 Data Collection**: Collects industrial metrics from Zabbix API based on search criteria
2. **🤖 Model Training**: Creates LSTM models compatible with your existing architecture  
3. **📡 Real-time Monitoring**: Continuous forecasting with live Zabbix data
4. **🛑 Clean Stopping**: Graceful shutdown mechanism with 'Q' key or system signals
5. **📊 Anomaly Detection**: Uses your existing forecasting and classification models

## 🚀 Quick Start

### Prerequisites

- Debian/Ubuntu system with Zabbix server already installed
- Python 3.7+ with pip
- Access to Zabbix web interface and API

### 1. Installation

```bash
# Clone or copy the zabbix_simple directory to your server
cd zabbix_simple

# Run setup script (as root for system-wide installation)
sudo ./setup.sh

# Or for user installation
./setup.sh
```

### 2. Configuration

Edit `config.json`:

```json
{
  "zabbix": {
    "url": "http://your-zabbix-server",
    "user": "Admin",
    "password": "your-password"
  },
  "data_collection": {
    "search_criteria": ["cpu", "memory", "network", "disk", "temperature"],
    "host_groups": ["Linux servers", "Zabbix servers"],
    "update_interval": 60,
    "history_hours": 48
  },
  "model": {
    "context_length": 60,
    "prediction_horizon": 6,
    "model_path": "./trained_model"
  },
  "monitoring": {
    "dashboard_port": 8050,
    "log_level": "INFO"
  }
}
```

### 3. Test Connection

```bash
python3 collect_data.py --test
```

### 4. Collect Historical Data

```bash
# Collect 48 hours of training data
python3 collect_data.py --collect --hours 48
```

### 5. Train Initial Model

```bash
# Train model using collected data
python3 train_model.py --data data/training_data_YYYYMMDD_HHMMSS.csv
```

### 6. Start Real-time Monitoring

```bash
# Start monitoring loop
python3 online_forecasting.py

# Press 'Q' + Enter to stop gracefully
```

## 📁 File Structure

```
zabbix_simple/
├── config.json              # Configuration file
├── collect_data.py          # Zabbix data collector
├── train_model.py           # Initial model training
├── online_forecasting.py    # Main forecasting loop
├── requirements.txt         # Python dependencies
├── setup.sh                # Installation script
├── README.md               # This file
├── data/                   # Collected training data (created)
├── temp_data/             # Temporary cycle data (created)
└── trained_model/         # Saved models and scalers (created)
```

## 🔧 System Components

### Data Collection (`collect_data.py`)
- **Search-based discovery**: Finds Zabbix items matching criteria
- **Host group filtering**: Focuses on specified device groups
- **Time series preparation**: Formats data for LSTM training
- **Data cleaning**: Handles missing values and outliers

### Model Training (`train_model.py`)
- **Compatible architecture**: Uses your existing LSTM model structure
- **Automated preprocessing**: Applies differencing and scaling
- **Training validation**: Tests model performance before saving
- **File compatibility**: Saves models compatible with existing system

### Real-time Forecasting (`online_forecasting.py`)
- **Continuous cycles**: Configurable update intervals (default 60 seconds)
- **Live data collection**: Fetches fresh data from Zabbix each cycle
- **Graceful shutdown**: Clean stopping with 'Q' key or system signals
- **Error handling**: Robust error recovery and logging
- **Temporary storage**: Saves cycle data for debugging/analysis

## ⚙️ Configuration Options

### Search Criteria
Configure what types of data to collect:
```json
"search_criteria": [
  "cpu", "memory", "network", "disk", "temperature",
  "bits sent", "bits received", "interface", "load"
]
```

### Host Groups
Target specific device groups:
```json
"host_groups": [
  "Linux servers", "Zabbix servers", "Network equipment",
  "Industrial", "SCADA", "PLC", "Critical systems"
]
```

### Model Parameters
```json
"model": {
  "context_length": 60,      # Input window size (minutes)
  "prediction_horizon": 6,   # Future steps to predict
  "model_path": "./trained_model"
}
```

## 🛑 Clean Stopping Mechanisms

The system supports multiple ways to stop gracefully:

1. **Interactive Mode**: Press 'Q' + Enter while running
2. **System Signals**: Send SIGINT (Ctrl+C) or SIGTERM
3. **Systemd**: `sudo systemctl stop zabbix-anomaly`

All methods ensure:
- Current prediction cycle completes
- Temporary data is saved
- Connections are closed properly
- Session statistics are logged

## 🧪 Testing and Troubleshooting

### Test Components

```bash
# Test Zabbix connection
python3 collect_data.py --test

# Test system initialization
python3 online_forecasting.py --test
```

### Common Issues

**"No hosts found"**
- Check host group names in config.json
- Verify user has access to specified groups
- Ensure hosts are enabled in Zabbix

**"No items found matching criteria"**
- Review search criteria in config.json
- Check if items are monitored and have numeric values
- Verify item names contain expected keywords

**"Model loading failed"**
- Ensure model was trained successfully
- Check file permissions in model directory
- Verify TensorFlow compatibility

### Debug Mode

Enable detailed logging:
```json
"monitoring": {
  "log_level": "DEBUG"
}
```

View logs:
```bash
tail -f forecasting_loop.log
```

## 🏭 Production Deployment

### System Service Installation

When installed as root, the setup script creates a systemd service:

```bash
# Enable and start service
sudo systemctl enable zabbix-anomaly
sudo systemctl start zabbix-anomaly

# Monitor service
sudo systemctl status zabbix-anomaly
sudo journalctl -u zabbix-anomaly -f

# Stop service
sudo systemctl stop zabbix-anomaly
```

### Service Configuration

The service is configured with:
- Automatic restart on failure
- Proper logging to syslog
- Clean shutdown handling
- Working directory management

## 📈 Performance Characteristics

- **Data Collection**: ~5-10 seconds per cycle
- **Prediction**: ~1-2 seconds per cycle  
- **Memory Usage**: ~200-500MB typical
- **CPU Usage**: ~5-15% during prediction
- **Network**: ~1-5MB per cycle (depends on item count)

## 🆚 Comparison with Complex Integration

| Feature | Complex System | Simple System |
|---------|----------------|---------------|
| **Files** | 20+ files | 4 core files |
| **Setup** | Multi-step manual | Single script |
| **Dependencies** | Many optional packages | Minimal required only |
| **Maintenance** | Complex monitoring | Single process |
| **Deployment** | Manual configuration | Automated service |
| **Stopping** | Force termination | Graceful shutdown |

## 🎯 Usage Examples

### Collect Weekly Training Data
```bash
python3 collect_data.py --collect --hours 168
```

### Train with Custom Parameters
```bash
python3 train_model.py --data data/training_data.csv --epochs 100 --batch-size 64
```

### Monitor with Different Update Interval
Edit config.json:
```json
"data_collection": {
  "update_interval": 30  // 30-second cycles
}
```

## 🔄 Integration Benefits

### For Research
- **Zero changes** to existing anomaly detection algorithms
- **Full compatibility** with trained models
- **Same accuracy** and performance characteristics
- **Preserved functionality** for continued development

### For Production
- **Live industrial data** from real Zabbix systems
- **Simplified deployment** and maintenance
- **Robust error handling** for 24/7 operation
- **Easy monitoring** and troubleshooting
- **Clean shutdown** for maintenance windows

---

## 📋 Step-by-Step Checklist

- [ ] Install on Debian system with Zabbix server
- [ ] Configure Zabbix connection in config.json
- [ ] Test connection: `python3 collect_data.py --test`
- [ ] Collect training data: `python3 collect_data.py --collect`
- [ ] Train model: `python3 train_model.py --data data/training_data_*.csv`
- [ ] Test system: `python3 online_forecasting.py --test`
- [ ] Start monitoring: `python3 online_forecasting.py`
- [ ] Verify clean stopping works: Press 'Q' + Enter

**Transform your research into production industrial monitoring!** 🏭✨