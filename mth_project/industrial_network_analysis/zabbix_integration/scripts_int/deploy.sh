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

# Install dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
apt update
apt install -y python3 python3-pip git

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
read -p "Enter Zabbix server URL: " ZABBIX_URL
read -p "Enter Zabbix username: " ZABBIX_USER
read -s -p "Enter Zabbix password: " ZABBIX_PASSWORD
echo ""

# Update config
sed -i "s|your-zabbix-server|${ZABBIX_URL#http://}|g" $INSTALL_DIR/config.json
sed -i "s|Admin|$ZABBIX_USER|g" $INSTALL_DIR/config.json
sed -i "s|your-password|$ZABBIX_PASSWORD|g" $INSTALL_DIR/config.json

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