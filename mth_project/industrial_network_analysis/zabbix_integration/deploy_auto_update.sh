#!/bin/bash

echo "🔄 Deploying auto-variable-update feature..."

# Copy the updated main.py
sudo cp main.py /opt/anomaly_detection/

echo "✅ Updated main.py with auto-variable feature"

# Test the new update feature
echo "🧪 Testing variable update..."
python3 /opt/anomaly_detection/main.py --update-variables

echo "🔄 Restarting service..."
sudo systemctl restart zabbix-monitoring

echo "✅ Deployment complete!"
echo "📊 Features:"
echo "   🔧 Auto-updates variables.txt when mismatch detected"
echo "   📝 Manual update: python3 /opt/anomaly_detection/main.py --update-variables"
echo "   🔍 Check logs: sudo journalctl -u zabbix-monitoring -f"