# 🔧 Zabbix Connection Troubleshooting Guide

Based on your test results, the **model loading is working correctly**, but the **Zabbix connection is failing**.

## 🎯 Your Current Status

✅ **TensorFlow/Model Loading**: Working (model loaded successfully)  
❌ **Zabbix Connection**: Failed  

## 🔍 Diagnostic Steps

### 1. Test Basic Zabbix Connection
Run the simple connection test:
```bash
cd /opt/anomaly_detection
python3 test_zabbix.py
```

This will show exactly what's failing in the connection.

### 2. Check Zabbix Server Status
```bash
# Check if Zabbix is running
systemctl status zabbix-server
systemctl status zabbix-web

# Check if web interface is accessible
curl -I http://localhost/zabbix
```

### 3. Verify Configuration
Check your `/opt/anomaly_detection/config.json`:

```json
{
  "zabbix": {
    "url": "http://localhost/zabbix",     ← Should match your Zabbix URL
    "user": "Admin",                      ← Zabbix username
    "password": "zabbix"                  ← Zabbix password
  },
  "monitoring": {
    "host_groups": ["Zabbix servers"]     ← Must match actual host group names
  }
}
```

### 4. Test Web Access
Try accessing Zabbix web interface:
```bash
# In browser or curl
http://localhost/zabbix
```

### 5. Common Issues & Solutions

#### Issue: Connection Refused
```
❌ Connection refused to localhost:80
```
**Solution**: Zabbix web server not running
```bash
systemctl start httpd    # or apache2
systemctl start zabbix-web
```

#### Issue: Authentication Failed
```
❌ Login failed for user 'Admin'
```
**Solutions**:
1. Check username/password in Zabbix web interface
2. Ensure user has **API access** enabled:
   - Login to Zabbix web → Administration → Users
   - Select your user → Permissions tab
   - Check "Frontend access" = Enabled

#### Issue: Wrong URL Format
```
❌ Invalid URL format
```
**Solutions**:
- Use: `http://localhost/zabbix` (with /zabbix)
- Not: `http://localhost` or `http://localhost:10051`

#### Issue: Host Group Not Found
```
⚠️ Not found: Zabbix servers
```
**Solutions**:
1. Check actual host group names in Zabbix web interface
2. Update config.json with correct names
3. Common group names: "Linux servers", "Zabbix servers", "Templates"

### 6. Quick Fixes to Try

#### Option A: Use Default Zabbix Setup
```json
{
  "zabbix": {
    "url": "http://localhost/zabbix",
    "user": "Admin",
    "password": "zabbix"
  },
  "monitoring": {
    "host_groups": ["Linux servers", "Zabbix servers"]
  }
}
```

#### Option B: Test with Templates Group
```json
{
  "monitoring": {
    "host_groups": ["Templates"]
  }
}
```

### 7. Detailed Connection Test

Run the enhanced test:
```bash
cd /opt/anomaly_detection
python3 zabbix_monitor.py --test
```

This will show:
- ✅ Zabbix connection status
- ✅ Available host groups  
- ✅ Hosts in your configured groups
- ✅ Sample data collection

## 🚀 Next Steps

1. **First**: Run `python3 test_zabbix.py` to see detailed connection info
2. **Fix config**: Update config.json based on test results  
3. **Retest**: Run `python3 zabbix_monitor.py --test`
4. **Start monitoring**: Once test passes, run `./start.sh`

## 💡 Expected Output When Working

```bash
🧪 Testing connections...
==================================================

1️⃣ Testing Zabbix Connection...
🔗 Attempting to connect to: http://localhost/zabbix
👤 Using user: Admin
🔐 Attempting login...
📡 Testing API connection...
✅ Connected to Zabbix 6.4.0

2️⃣ Testing Model Loading...
✅ Model loaded

3️⃣ Testing Data Collection...
📊 Sample data collected: (120, 15)
🏷️ Sample variables: ['host1_cpu', 'host1_memory', 'host2_cpu', 'host2_memory', 'host3_network']...
✅ All systems ready!
==================================================
```

Your **anomaly detection models are working perfectly** - we just need to get the Zabbix connection configured correctly! 🏭✨