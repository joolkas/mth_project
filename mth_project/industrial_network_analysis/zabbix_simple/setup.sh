#!/bin/bash
# Simple Zabbix Integration Setup Script
# For Debian/Ubuntu systems where Zabbix server is already installed

set -e

echo "🏭 Setting up Simple Zabbix Integration for Industrial Network Anomaly Detection"
echo "================================================================================"

# Check if running as root for system-wide installation
if [[ $EUID -eq 0 ]]; then
    echo "ℹ️  Running as root - will install system-wide"
    INSTALL_DIR="/opt/zabbix_anomaly_detection"
    SERVICE_INSTALL=true
else
    echo "ℹ️  Running as user - will install locally"
    INSTALL_DIR="$HOME/zabbix_anomaly_detection"
    SERVICE_INSTALL=false
fi

# Save current directory (source directory)
SOURCE_DIR="$(pwd)"

# Create installation directory
echo "📁 Creating installation directory: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

# Copy files from source directory
echo "📋 Copying program files..."
cp "$SOURCE_DIR"/*.py "$INSTALL_DIR/"
cp "$SOURCE_DIR"/*.json "$INSTALL_DIR/"
cp "$SOURCE_DIR"/requirements.txt "$INSTALL_DIR/"

# Change to installation directory for remaining operations
cd "$INSTALL_DIR"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python 3.7+"
    exit 1
fi

# Install pip if not available
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "📦 Installing pip..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3-pip
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3-pip
    else
        echo "❌ Cannot install pip automatically. Please install manually."
        exit 1
    fi
fi

# Install requirements
echo "📦 Installing Python packages..."
$PYTHON_CMD -m pip install --user -r requirements.txt

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data temp_data trained_model

# Set permissions
if [[ $EUID -eq 0 ]]; then
    chown -R root:root "$INSTALL_DIR"
    chmod -R 755 "$INSTALL_DIR"
    chmod +x "$INSTALL_DIR"/*.py
fi

# Create systemd service if running as root
if [[ $SERVICE_INSTALL == true ]]; then
    echo "🔧 Creating systemd service..."
    cat > /etc/systemd/system/zabbix-anomaly.service << EOF
[Unit]
Description=Zabbix Industrial Network Anomaly Detection
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=$PYTHON_CMD online_forecasting.py
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=zabbix-anomaly

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    echo "✅ Systemd service created: zabbix-anomaly.service"
fi

echo ""
echo "✅ Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit config.json with your Zabbix server details:"
echo "   - Zabbix server URL"
echo "   - Username and password"
echo "   - Host groups to monitor"
echo ""
echo "2. Test the connection:"
echo "   cd $INSTALL_DIR"
echo "   $PYTHON_CMD collect_data.py --test"
echo ""
echo "3. Collect training data:"
echo "   $PYTHON_CMD collect_data.py --collect --hours 48"
echo ""
echo "4. Train the initial model:"
echo "   $PYTHON_CMD train_model.py --data data/training_data_YYYYMMDD_HHMMSS.csv"
echo ""
echo "5. Start real-time monitoring:"
echo "   $PYTHON_CMD online_forecasting.py"
echo ""

if [[ $SERVICE_INSTALL == true ]]; then
    echo "🔧 System service commands:"
    echo "   Start service: sudo systemctl start zabbix-anomaly"
    echo "   Enable on boot: sudo systemctl enable zabbix-anomaly"
    echo "   View logs: sudo journalctl -u zabbix-anomaly -f"
    echo "   Stop service: sudo systemctl stop zabbix-anomaly"
fi

echo ""
echo "🎯 The system is now ready for industrial network anomaly detection!"