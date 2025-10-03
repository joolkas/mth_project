#!/bin/bash

# Update variables.txt with current Zabbix variables
echo "🔄 Updating variables.txt with current Zabbix variables..."

# Copy the new variables file
sudo cp variables.txt /opt/anomaly_detection/models/forecasting_model/

# Verify the file was copied
echo "📁 Variables file updated:"
sudo cat /opt/anomaly_detection/models/forecasting_model/variables.txt | wc -l
echo "   (Should show 23 lines)"

# Restart the service to pick up the new variables
echo "🔄 Restarting zabbix-monitoring service..."
sudo systemctl restart zabbix-monitoring

echo "✅ Done! Check logs with: sudo journalctl -u zabbix-monitoring -f"