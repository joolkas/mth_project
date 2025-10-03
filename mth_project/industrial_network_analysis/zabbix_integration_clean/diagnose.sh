#!/bin/bash

echo "🔍 Zabbix Anomaly Detection Service Diagnostics"
echo "================================================"

# Check if service is running
echo "1️⃣ Service Status:"
systemctl status zabbix-anomaly-monitor --no-pager

echo -e "\n2️⃣ Recent Service Logs:"
journalctl -u zabbix-anomaly-monitor --no-pager -n 50

echo -e "\n3️⃣ Check if process is running:"
ps aux | grep zabbix_monitor

echo -e "\n4️⃣ Check ports (Dash should be on 8050):"
netstat -tulpn | grep :8050 || echo "Port 8050 not in use"

echo -e "\n5️⃣ Check log files:"
if [ -f "/opt/anomaly_detection/zabbix_monitor.log" ]; then
    echo "Recent log entries:"
    tail -20 /opt/anomaly_detection/zabbix_monitor.log
else
    echo "No log file found at /opt/anomaly_detection/zabbix_monitor.log"
fi

echo -e "\n6️⃣ Test configuration:"
cd /opt/anomaly_detection
python3 -c "
import json
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    print('✅ Config file is valid JSON')
    print(f'   Zabbix URL: {config[\"zabbix\"][\"url\"]}')
    print(f'   Model path: {config[\"models\"][\"forecasting_model_path\"]}')
    print(f'   Device groups: {config[\"industrial_filters\"][\"device_groups\"]}')
except Exception as e:
    print(f'❌ Config error: {e}')
"

echo -e "\n7️⃣ Test basic functionality:"
python3 zabbix_monitor.py --test

echo -e "\n================================================"
echo "🚀 Troubleshooting Tips:"
echo "   - If service is 'failed': Check logs above"
echo "   - If service is 'active' but no logs: Check permissions"
echo "   - If port 8050 not open: Dash server startup failed"
echo "   - Try running manually: cd /opt/anomaly_detection && python3 zabbix_monitor.py"