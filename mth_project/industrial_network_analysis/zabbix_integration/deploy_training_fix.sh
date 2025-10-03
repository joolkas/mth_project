#!/bin/bash

echo "🔧 Deploying training data collection fix..."

# Copy the updated get_data_real_system.py
sudo cp get_data_real_system.py /opt/anomaly_detection/

echo "✅ Updated get_data_real_system.py with current Zabbix data format"

# Test the data collection
echo "🧪 Testing data collection..."
cd /opt/anomaly_detection
python3 get_data_real_system.py

echo "✅ Fix deployed!"
echo ""
echo "📊 What this fixes:"
echo "   ✅ Works with current Zabbix data format (42 features)"
echo "   ✅ Uses Dataset class for consistent preprocessing"  
echo "   ✅ Collects 7 days of data for better training"
echo "   ✅ Creates proper time series format"
echo ""
echo "🎯 Next: Use the generated CSV files to train your model with 42 features"