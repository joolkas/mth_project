#!/bin/bash

echo "🏭 Simple Zabbix Monitoring - Deployment"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run as root: sudo $0"
    exit 1
fi

# Install basic requirements
echo "📦 Installing requirements..."
pip3 install pandas numpy scikit-learn tensorflow pyzabbix plotly dash

# Create directories
INSTALL_DIR="/opt/anomaly_detection"
mkdir -p $INSTALL_DIR/models

# Copy simple files
echo "📋 Installing simplified system..."
cp simple_main.py $INSTALL_DIR/main.py
cp simple_training.py $INSTALL_DIR/
cp config.json $INSTALL_DIR/

# Copy supporting files
cp ../data_utils.py $INSTALL_DIR/
cp ../initial_model.py $INSTALL_DIR/  
cp ../online_forecasting_multi_step.py $INSTALL_DIR/
cp ../data_preprocessing.py $INSTALL_DIR/

# Copy model if exists
if [ -d "../forecasting_model" ]; then
    cp -r ../forecasting_model $INSTALL_DIR/models/
    echo "✅ Model copied"
else
    echo "⚠️ No model found - you'll need to train one first"
fi

# Create simplified systemd service
echo "🔧 Creating service..."
cat > /etc/systemd/system/zabbix-monitoring.service << EOF
[Unit]
Description=Simple Zabbix Industrial Monitoring
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/python3 $INSTALL_DIR/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Set permissions
chown -R root:root $INSTALL_DIR
chmod +x $INSTALL_DIR/main.py
chmod +x $INSTALL_DIR/simple_training.py

# Enable service
systemctl daemon-reload
systemctl enable zabbix-monitoring

echo ""
echo "✅ Simple deployment complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Test connection: python3 $INSTALL_DIR/main.py --test"
echo "2. Collect training data: python3 $INSTALL_DIR/simple_training.py"
echo "3. Train your model with the collected data"
echo "4. Start monitoring: systemctl start zabbix-monitoring"
echo "5. View logs: journalctl -u zabbix-monitoring -f"
echo ""
echo "📊 Features:"
echo "✅ Auto-discovers all Zabbix hosts and metrics"
echo "✅ Auto-adapts to any number of features"
echo "✅ Auto-saves current variables"
echo "✅ Simple error handling and recovery"
echo "✅ Works with any Zabbix environment"