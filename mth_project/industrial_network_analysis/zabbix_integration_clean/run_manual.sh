#!/bin/bash

echo "🚀 Manual Zabbix Monitor Test"
echo "============================="

cd /opt/anomaly_detection

echo "📁 Current directory: $(pwd)"
echo "📋 Files present:"
ls -la

echo -e "\n🔧 Config check:"
python3 -c "
import json
with open('config.json', 'r') as f:
    config = json.load(f)
print('✅ Config loaded successfully')
print(f'Zabbix URL: {config[\"zabbix\"][\"url\"]}')
print(f'Device groups: {config[\"industrial_filters\"][\"device_groups\"]}')
"

echo -e "\n🧪 Quick connection test:"
python3 test_zabbix.py

echo -e "\n🚀 Starting monitor in foreground mode..."
echo "   (This will show all logs directly)"
echo "   Press Ctrl+C to stop"
echo "   Dashboard: http://localhost:8050"
echo ""

# Run in foreground so we can see all output
python3 zabbix_monitor.py