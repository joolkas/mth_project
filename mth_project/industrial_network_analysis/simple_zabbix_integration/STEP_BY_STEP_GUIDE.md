# 🚀 Step-by-Step Guide: Running Your Simplified Zabbix Integration

## 📍 **Current Location**
You are here: `simple_zabbix_integration/` directory

## 🎯 **Step 1: Install Missing Dependencies**

### Option A: Install PyZabbix (Required)
```powershell
pip install pyzabbix
```

### Option B: Install All Dependencies (Recommended)
```powershell
pip install -r requirements.txt
```

### ✅ **Verify Installation**
```powershell
python test_standalone.py
```
**Expected Result**: All 5 tests should pass ✅

---

## 🔧 **Step 2: Configure Your Zabbix Connection**

### Edit `config.json` with your Zabbix details:
```json
{
    "zabbix": {
        "url": "http://YOUR_ZABBIX_SERVER/zabbix",
        "username": "YOUR_USERNAME", 
        "password": "YOUR_PASSWORD"
    }
}
```

### 🔍 **Example Configuration**
Replace these values with your actual Zabbix server details:
- `YOUR_ZABBIX_SERVER`: Your Zabbix server IP/hostname
- `YOUR_USERNAME`: Your Zabbix login username  
- `YOUR_PASSWORD`: Your Zabbix login password

---

## 🧪 **Step 3: Test Zabbix Connection**

```powershell
python get_data.py --test
```

**Expected Result**: 
```
✅ Connected to Zabbix successfully
✅ Found X hosts
✅ Found Y network items
```

**If connection fails:**
- Check your Zabbix server URL
- Verify username/password
- Ensure Zabbix server is accessible

---

## 📊 **Step 4: Collect Training Data**

### Collect 48 hours of historical data:
```powershell
python get_data.py --collect-history --hours 48
```

**Expected Result:**
```
📊 Collecting data for 48 hours...
✅ Found X network items
📈 Collected Y data points
💾 Saved to: data/historical_data_YYYYMMDD_HHMMSS.csv
```

### 🔍 **Check Your Data**
```powershell
dir data\
```
You should see a CSV file with your collected data.

---

## 🤖 **Step 5: Train Your Model**

```powershell
python train_model.py --data data\historical_data_*.csv
```

**Expected Result:**
```
🤖 Training LSTM model...
📊 Data shape: (samples, 60, features)
🔄 Training progress: [====>] 100%
✅ Model trained successfully
💾 Model saved to: trained_model/lstm_model.h5
```

### 🔍 **Verify Model Creation**
```powershell
dir trained_model\
```
You should see your trained model files.

---

## 🚀 **Step 6: Start Real-Time Monitoring**

```powershell
python main_forecasting.py
```

**Expected Result:**
```
🚀 Starting real-time forecasting...
🌐 Dashboard starting at: http://localhost:8050
🔄 Monitoring every 60 seconds...
📊 Predictions updated...
```

### 🌐 **View Dashboard**
Open your browser and go to: **http://localhost:8050**

---

## 🎛️ **Step 7: Monitor the System**

### What You'll See:
1. **Terminal Output**: Live predictions and status updates
2. **Web Dashboard**: Real-time graphs and metrics at http://localhost:8050
3. **Data Files**: New predictions saved in `temp_data/`

### 🔄 **System Cycle** (Every 60 seconds):
1. Collect latest data from Zabbix
2. Make 6-step forecasting predictions  
3. Update dashboard with new results
4. Check for anomalies
5. Log results

---

## 🛠️ **Troubleshooting Guide**

### ❌ **Problem**: "No module named 'pyzabbix'"
**Solution**: 
```powershell
pip install pyzabbix
```

### ❌ **Problem**: "Connection to Zabbix failed"
**Solutions**:
1. Check `config.json` Zabbix URL format
2. Verify username/password
3. Test Zabbix web interface manually
4. Check network connectivity

### ❌ **Problem**: "No data collected"
**Solutions**:
1. Check your search criteria in `config.json`
2. Verify hosts have network monitoring items
3. Check time range (try shorter period first)

### ❌ **Problem**: "Model training failed"
**Solutions**:
1. Ensure you have collected data first
2. Check CSV file is not empty
3. Verify TensorFlow installation

### ❌ **Problem**: "Dashboard not loading"
**Solutions**:
1. Check port 8050 is not in use
2. Try different port in config
3. Check firewall settings

---

## 📋 **Quick Command Reference**

```powershell
# Test system
python test_standalone.py

# Test Zabbix connection
python get_data.py --test

# Collect training data (48 hours)
python get_data.py --collect-history --hours 48

# Train model
python train_model.py --data data\historical_data_*.csv

# Start monitoring
python main_forecasting.py

# Check system status
dir data\
dir trained_model\
dir temp_data\
```

---

## 🎯 **Success Indicators**

### ✅ **System is Working When You See**:
1. **Zabbix connection**: "Connected successfully"
2. **Data collection**: CSV files in `data/` folder
3. **Model training**: Model files in `trained_model/` folder  
4. **Real-time monitoring**: Dashboard at http://localhost:8050
5. **Predictions**: Regular updates every 60 seconds

### 🎉 **You're Done!**
Your industrial network anomaly detection system is now running with:
- ✅ Real-time Zabbix data collection
- ✅ LSTM forecasting every 60 seconds  
- ✅ Web dashboard visualization
- ✅ Anomaly classification integration

---

## 📞 **Need Help?**

If you encounter issues:
1. Check the terminal output for error messages
2. Verify each step completed successfully
3. Review the troubleshooting section above
4. Check your Zabbix server is accessible and configured correctly