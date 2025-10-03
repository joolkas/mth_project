#!/bin/bash

# 🏭 Simple Deployment Script for Zabbix Industrial Anomaly Detection

echo "🚀 Deploying Zabbix Industrial Anomaly Detection System"

# Check if running as root for system installation
if [ "$EUID" -ne 0 ]; then
    echo "⚠️ Running as regular user - will install to current directory"
    INSTALL_DIR="./anomaly_detection"
    USE_SYSTEMD=false
else
    echo "🔧 Running as root - will install system-wide"
    INSTALL_DIR="/opt/anomaly_detection"
    USE_SYSTEMD=true
fi

# Create installation directory
echo "📁 Creating installation directory: $INSTALL_DIR"
mkdir -p $INSTALL_DIR
mkdir -p $INSTALL_DIR/models

# Install Python requirements
echo "📦 Installing Python requirements..."
pip3 install -r requirements.txt

# Copy main files
echo "📋 Installing system files..."
cp zabbix_monitor.py $INSTALL_DIR/
cp test_zabbix.py $INSTALL_DIR/
cp config.json $INSTALL_DIR/
cp requirements.txt $INSTALL_DIR/

# Make executable
chmod +x $INSTALL_DIR/zabbix_monitor.py
chmod +x $INSTALL_DIR/test_zabbix.py

# Copy model if it exists
if [ -d "../forecasting_model" ]; then
    echo "🤖 Copying trained model..."
    cp -r ../forecasting_model $INSTALL_DIR/models/
else
    echo "⚠️ No trained model found at ../forecasting_model"
    echo "   Please copy your trained model to $INSTALL_DIR/models/forecasting_model"
fi

# Create systemd service if running as root
if $USE_SYSTEMD; then
    echo "⚙️ Creating systemd service..."
    
    cat > /etc/systemd/system/zabbix-anomaly-monitor.service << EOF
[Unit]
Description=Zabbix Industrial Anomaly Detection Monitor
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
Environment=PYTHONPATH=$INSTALL_DIR:$INSTALL_DIR/../industrial_network_analysis
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 $INSTALL_DIR/zabbix_monitor.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable zabbix-anomaly-monitor
    
    echo "📋 Systemd service created: zabbix-anomaly-monitor"
    echo "   Start with: sudo systemctl start zabbix-anomaly-monitor"
    echo "   View logs: sudo journalctl -u zabbix-anomaly-monitor -f"
fi

# Create simple start script
cat > $INSTALL_DIR/start.sh << 'EOF'
#!/bin/bash
echo "🏭 Starting Zabbix Industrial Anomaly Detection"
echo "   Dashboard: http://localhost:8050"
echo "   Press Ctrl+C to stop"
echo ""
python3 zabbix_monitor.py
EOF

chmod +x $INSTALL_DIR/start.sh

# Create test script
cat > $INSTALL_DIR/test.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing Zabbix connection and models..."
python3 zabbix_monitor.py --test
EOF

chmod +x $INSTALL_DIR/test.sh

echo ""
echo "✅ Installation completed!"
echo ""
echo "📁 Installation directory: $INSTALL_DIR"
echo ""
echo "🔧 Next steps:"
echo "   1. Edit $INSTALL_DIR/config.json with your Zabbix credentials"
echo "   2. Ensure your trained model is in $INSTALL_DIR/models/forecasting_model/"
echo "   3. Test connection: $INSTALL_DIR/test.sh"
echo ""

if $USE_SYSTEMD; then
    echo "🚀 Start monitoring:"
    echo "   sudo systemctl start zabbix-anomaly-monitor"
    echo ""
    echo "📊 View logs:"
    echo "   sudo journalctl -u zabbix-anomaly-monitor -f"
else
    echo "🚀 Start monitoring:"
    echo "   cd $INSTALL_DIR && ./start.sh"
fi

echo ""
echo "🌐 Dashboard will be available at: http://localhost:8050"
echo ""