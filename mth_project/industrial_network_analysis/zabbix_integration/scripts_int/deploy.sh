#!/bin/bash
# Simple deployment script for Zabbix Industrial Monitoring

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🏭 Simple Zabbix Industrial Monitoring - Deployment${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}❌ Please run as root: sudo $0${NC}"
    exit 1
fi

# Check if dependencies are already installed
echo -e "${BLUE}📦 Checking dependencies...${NC}"

if command -v python3 >/dev/null 2>&1 && command -v pip3 >/dev/null 2>&1 && command -v git >/dev/null 2>&1; then
    echo "✅ Dependencies already installed (python3, pip3, git)"
else
    echo "Installing missing dependencies..."
    
    if command -v apt >/dev/null 2>&1; then
    # Debian/Ubuntu
    echo "Detected Debian/Ubuntu system"
    apt update
    apt install -y python3 python3-pip git
elif command -v dnf >/dev/null 2>&1; then
    # RHEL 9/CentOS Stream 9/Rocky Linux 9/Fedora
    echo "Detected system with dnf (RHEL 9/CentOS/Rocky/Fedora)"
    
    # Check if RHEL and enable EPEL correctly
    if grep -q "Red Hat Enterprise Linux" /etc/os-release; then
        echo "RHEL detected - enabling EPEL..."
        dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm 2>/dev/null || true
    fi
    
    dnf install -y python3 python3-pip git
elif command -v yum >/dev/null 2>&1; then
    # Legacy CentOS/RHEL 7
    echo "Detected legacy CentOS/RHEL 7 system"
    yum install -y epel-release
    yum install -y python3 python3-pip git
    else
        echo -e "${RED}❌ Unsupported Linux distribution${NC}"
        echo "Please install manually: python3, python3-pip, git"
        exit 1
    fi
fi

# Install Python packages
pip3 install -r requirements.txt

# Create directories
INSTALL_DIR="/opt/anomaly_detection"
mkdir -p $INSTALL_DIR/models

# Copy files
echo -e "${BLUE}📋 Installing files...${NC}"
cp main.py $INSTALL_DIR/
cp config.json $INSTALL_DIR/

# Copy your trained models (user needs to do this)
echo -e "${BLUE}📦 Model setup...${NC}"
echo "IMPORTANT: Copy your trained models to $INSTALL_DIR/models/"
echo "Expected: $INSTALL_DIR/models/forecasting_model/"
echo ""

# Get configuration
# read -p "Enter Zabbix server URL: " ZABBIX_URL
# read -p "Enter Zabbix username: " ZABBIX_USER
# read -s -p "Enter Zabbix password: " ZABBIX_PASSWORD
echo ""

# Update config
# sed -i "s|your-zabbix-server|${ZABBIX_URL#https://}|g" $INSTALL_DIR/config.json
# sed -i "s|Admin|$ZABBIX_USER|g" $INSTALL_DIR/config.json
# sed -i "s|your-password|$ZABBIX_PASSWORD|g" $INSTALL_DIR/config.json

cp ../data_utils.py $INSTALL_DIR/
cp ../initial_model.py $INSTALL_DIR/  
cp ../online_forecasting_multi_step.py $INSTALL_DIR/
cp ../dash_plotter.py $INSTALL_DIR/
cp ../get_data.py $INSTALL_DIR/
cp ../data_preprocessing.py $INSTALL_DIR/
cp -r ../forecasting_model $INSTALL_DIR/models/ 2>/dev/null || echo "Manual model copy needed"

# Create systemd service
echo -e "${BLUE}🔧 Creating service...${NC}"
cat > /etc/systemd/system/zabbix-monitoring.service << EOF
[Unit]
Description=Zabbix Industrial Monitoring
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/python3 $INSTALL_DIR/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Set permissions
chown -R root:root $INSTALL_DIR
chmod +x $INSTALL_DIR/main.py

# Enable service
systemctl daemon-reload
systemctl enable zabbix-monitoring

echo ""
echo -e "${GREEN}✅ Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Copy your forecasting_model to: $INSTALL_DIR/models/"
echo "2. Start service: systemctl start zabbix-monitoring"
echo "3. Check status: systemctl status zabbix-monitoring"
echo "4. View dashboard: http://localhost:8050"
echo ""
echo "Test connection: python3 $INSTALL_DIR/main.py --test"