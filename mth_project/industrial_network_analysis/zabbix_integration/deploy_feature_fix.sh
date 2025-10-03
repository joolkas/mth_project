#!/bin/bash

echo "🔧 Deploying feature selection fix..."

# Copy the updated main.py with feature selection
sudo cp main.py /opt/anomaly_detection/

echo "✅ Updated main.py with auto-feature-selection"

# Restart the service to apply the fix
echo "🔄 Restarting zabbix-monitoring service..."
sudo systemctl restart zabbix-monitoring

echo "✅ Fix deployed!"
echo ""
echo "📊 What this fixes:"
echo "   🎯 Automatically selects 23 most important features from 42 available"
echo "   🧠 Prioritizes: memory, CPU, network, disk metrics"
echo "   ✅ Model input shape now matches: (None, 60, 23)"
echo ""
echo "🔍 Monitor logs: sudo journalctl -u zabbix-monitoring -f"