# 🔧 Quick Config Fix

Your current config has:
```json
{
  "zabbix": {
    "url": "https://192.168.93.45",
    "user": "Admin",
    "password": "zabbix"
  }
}
```

## 🎯 Try These URL Formats

### Option 1: Add /zabbix path
```json
{
  "zabbix": {
    "url": "https://192.168.93.45/zabbix",
    "user": "Admin",
    "password": "zabbix"
  }
}
```

### Option 2: If using custom port
```json
{
  "zabbix": {
    "url": "https://192.168.93.45:443/zabbix",
    "user": "Admin",
    "password": "zabbix"
  }
}
```

### Option 3: If no SSL (HTTP)
```json
{
  "zabbix": {
    "url": "http://192.168.93.45/zabbix",
    "user": "Admin",
    "password": "zabbix"
  }
}
```

## 🧪 Test Each Option

```bash
# Test with current config
cd /opt/anomaly_detection
python3 test_zabbix.py

# Try accessing web interface directly
curl -I https://192.168.93.45/zabbix
```

Most Zabbix installations need the `/zabbix` path in the URL!