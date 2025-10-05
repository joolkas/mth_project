#!/bin/bash

# 🏭 RHEL Production Setup for Simplified Zabbix Integration
# Compatible with RHEL 7, 8, and 9

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="zabbix-anomaly-detection"
SERVICE_USER="anomaly"

# Default installation directory - can be overridden
INSTALL_DIR="/opt/zabbix_anomaly_detection"

# Check if custom installation directory provided
if [ "$1" != "" ]; then
    INSTALL_DIR="$1"
    echo "🔧 Using custom installation directory: $INSTALL_DIR"
fi

echo "🏭 RHEL Production Setup - Zabbix Industrial Anomaly Detection"
echo "=============================================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run as root (use sudo)"
    echo "   Usage: sudo ./rhel_install.sh"
    exit 1
fi

# Detect RHEL version
if [ -f /etc/redhat-release ]; then
    RHEL_VERSION=$(cat /etc/redhat-release | grep -oE '[0-9]+' | head -1)
    echo "🔍 Detected RHEL version: $RHEL_VERSION"
else
    echo "⚠️  Could not detect RHEL version, assuming RHEL 8"
    RHEL_VERSION=8
fi

# Function to install EPEL repository
install_epel() {
    echo "📦 Installing EPEL repository..."
    case $RHEL_VERSION in
        7)
            yum install -y epel-release
            ;;
        8|9)
            dnf install -y epel-release
            ;;
        *)
            echo "⚠️  Unknown RHEL version, trying dnf..."
            dnf install -y epel-release || yum install -y epel-release
            ;;
    esac
}

# Function to install system packages
install_system_packages() {
    echo "📦 Installing system packages..."
    
    # Common packages for all RHEL versions
    PACKAGES="python3 python3-pip python3-devel gcc gcc-c++ make openssl-devel libffi-devel"
    
    case $RHEL_VERSION in
        7)
            # RHEL 7 specific packages
            yum update -y
            yum groupinstall -y "Development Tools"
            yum install -y $PACKAGES python3-setuptools
            ;;
        8)
            # RHEL 8 specific packages
            dnf update -y
            dnf groupinstall -y "Development Tools"
            dnf install -y $PACKAGES python3-setuptools python3-wheel
            # Enable PowerTools repository for additional packages
            dnf config-manager --set-enabled powertools || true
            ;;
        9)
            # RHEL 9 specific packages
            dnf update -y
            dnf groupinstall -y "Development Tools"
            dnf install -y $PACKAGES python3-setuptools python3-wheel
            # Enable CRB repository for additional packages
            dnf config-manager --set-enabled crb || true
            ;;
        *)
            echo "⚠️  Unknown RHEL version, using dnf..."
            dnf update -y || yum update -y
            dnf install -y $PACKAGES || yum install -y $PACKAGES
            ;;
    esac
}

# Function to create service user
create_service_user() {
    echo "👤 Creating service user..."
    
    if ! id "$SERVICE_USER" &>/dev/null; then
        useradd -r -s /bin/false -d $INSTALL_DIR $SERVICE_USER
        echo "✅ Created user: $SERVICE_USER"
    else
        echo "✅ User $SERVICE_USER already exists"
    fi
}

# Function to install Python packages
install_python_packages() {
    echo "🐍 Installing Python packages..."
    
    # Upgrade pip first
    python3 -m pip install --upgrade pip
    
    # Install packages with specific versions for RHEL compatibility
    python3 -m pip install \
        pyzabbix>=1.0.0 \
        pandas>=1.3.0,<2.0.0 \
        numpy>=1.21.0,<2.0.0 \
        scikit-learn>=1.0.0,<1.4.0 \
        tensorflow>=2.8.0,<2.15.0 \
        plotly>=5.0.0 \
        dash>=2.0.0,<2.15.0 \
        urllib3>=1.26.0,<2.0.0 \
        werkzeug>=2.0.0,<3.0.0
    
    echo "✅ Python packages installed"
}

# Function to setup application directory
setup_application() {
    echo "📁 Setting up application directory..."
    
    # Create installation directory
    mkdir -p $INSTALL_DIR
    mkdir -p $INSTALL_DIR/data
    mkdir -p $INSTALL_DIR/temp_data
    mkdir -p $INSTALL_DIR/trained_model
    mkdir -p $INSTALL_DIR/logs
    
    # Copy application files
    cp "$SCRIPT_DIR"/*.py $INSTALL_DIR/
    cp "$SCRIPT_DIR"/config.json $INSTALL_DIR/
    cp "$SCRIPT_DIR"/requirements.txt $INSTALL_DIR/
    cp "$SCRIPT_DIR"/README.md $INSTALL_DIR/
    
    # Set permissions
    chown -R $SERVICE_USER:$SERVICE_USER $INSTALL_DIR
    chmod -R 755 $INSTALL_DIR
    chmod 644 $INSTALL_DIR/*.py $INSTALL_DIR/*.json $INSTALL_DIR/*.txt $INSTALL_DIR/*.md
    chmod 755 $INSTALL_DIR/*.py  # Make Python files executable
    
    echo "✅ Application directory setup complete"
}

# Function to create systemd service
create_systemd_service() {
    echo "⚙️  Creating systemd service..."
    
    cat > /etc/systemd/system/${SERVICE_NAME}.service << EOF
[Unit]
Description=Zabbix Industrial Anomaly Detection System
Documentation=file://$INSTALL_DIR/README.md
After=network-online.target
Wants=network-online.target
Requires=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
Environment=PYTHONPATH=$INSTALL_DIR
Environment=PYTHONUNBUFFERED=1

# Main service command
ExecStart=/usr/bin/python3 $INSTALL_DIR/main_forecasting.py

# Service management
Restart=always
RestartSec=10
StartLimitInterval=60
StartLimitBurst=3

# Process limits
LimitNOFILE=65536
LimitNPROC=4096

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$INSTALL_DIR

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable $SERVICE_NAME
    
    echo "✅ Systemd service created and enabled"
}

# Function to create log rotation
setup_log_rotation() {
    echo "📝 Setting up log rotation..."
    
    cat > /etc/logrotate.d/$SERVICE_NAME << EOF
$INSTALL_DIR/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 $SERVICE_USER $SERVICE_USER
    postrotate
        systemctl reload $SERVICE_NAME > /dev/null 2>&1 || true
    endscript
}
EOF

    echo "✅ Log rotation configured"
}

# Function to setup firewall
setup_firewall() {
    echo "🔥 Configuring firewall..."
    
    # Check if firewall is running
    if systemctl is-active --quiet firewalld; then
        echo "📡 Opening port 8050 for dashboard..."
        firewall-cmd --permanent --add-port=8050/tcp
        firewall-cmd --reload
        echo "✅ Firewall configured"
    else
        echo "⚠️  Firewalld not running, skipping firewall configuration"
        echo "   Manual step: Open port 8050 for dashboard access"
    fi
}

# Function to create helper scripts
create_helper_scripts() {
    echo "🔧 Creating helper scripts..."
    
    # Create start script
    cat > $INSTALL_DIR/start.sh << 'EOF'
#!/bin/bash
echo "🏭 Starting Zabbix Industrial Anomaly Detection"
sudo systemctl start zabbix-anomaly-detection
sudo systemctl status zabbix-anomaly-detection
echo "📊 Dashboard: http://localhost:8050"
EOF

    # Create stop script
    cat > $INSTALL_DIR/stop.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping Zabbix Industrial Anomaly Detection"
sudo systemctl stop zabbix-anomaly-detection
sudo systemctl status zabbix-anomaly-detection
EOF

    # Create status script
    cat > $INSTALL_DIR/status.sh << 'EOF'
#!/bin/bash
echo "📊 Zabbix Anomaly Detection System Status"
echo "========================================"
sudo systemctl status zabbix-anomaly-detection
echo ""
echo "📝 Recent logs:"
sudo journalctl -u zabbix-anomaly-detection -n 20 --no-pager
EOF

    # Create test script
    cat > $INSTALL_DIR/test.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing Zabbix Anomaly Detection System"
echo "=========================================="
cd /opt/zabbix_anomaly_detection
sudo -u anomaly python3 get_data.py --test
EOF

    # Make scripts executable
    chmod 755 $INSTALL_DIR/*.sh
    
    echo "✅ Helper scripts created"
}

# Function to create SELinux policy (if SELinux is enabled)
setup_selinux() {
    if command -v getenforce >/dev/null 2>&1 && [ "$(getenforce)" != "Disabled" ]; then
        echo "🔒 Configuring SELinux..."
        
        # Allow the service to bind to port 8050
        setsebool -P httpd_can_network_connect 1
        semanage port -a -t http_port_t -p tcp 8050 2>/dev/null || true
        
        # Set SELinux context for application directory
        semanage fcontext -a -t bin_t "$INSTALL_DIR/.*\.py" 2>/dev/null || true
        restorecon -R $INSTALL_DIR
        
        echo "✅ SELinux configured"
    else
        echo "ℹ️  SELinux not enabled, skipping SELinux configuration"
    fi
}

# Function to check disk space requirements
check_disk_space() {
    echo "💾 Checking disk space requirements..."
    
    # Check installation directory space
    INSTALL_DIR_PARENT=$(dirname "$INSTALL_DIR")
    INSTALL_AVAIL_GB=$(df "$INSTALL_DIR_PARENT" 2>/dev/null | awk 'NR==2 {print int($4/1024/1024)}' || df / | awk 'NR==2 {print int($4/1024/1024)}')
    
    echo "📁 Installation directory: $INSTALL_DIR"
    echo "💾 Available space: ${INSTALL_AVAIL_GB}GB"
    
    if [ "$INSTALL_AVAIL_GB" -lt 2 ]; then
        echo "❌ Insufficient disk space for installation"
        echo "   Minimum 2GB required, found ${INSTALL_AVAIL_GB}GB"
        echo ""
        echo "💡 Alternative installation locations with more space:"
        df -h | grep -E "/(home|var|tmp)" | while read -r line; do
            mount_point=$(echo "$line" | awk '{print $6}')
            avail_space=$(echo "$line" | awk '{print $4}')
            echo "   $mount_point ($avail_space available)"
        done
        echo ""
        echo "🔧 To install in a different location:"
        echo "   sudo $0 /home/zabbix_anomaly_detection"
        echo "   sudo $0 /var/opt/zabbix_anomaly_detection"
        exit 1
    elif [ "$INSTALL_AVAIL_GB" -lt 5 ]; then
        echo "⚠️  Limited disk space detected (${INSTALL_AVAIL_GB}GB available)"
        echo "   Installation will proceed, but monitor disk usage"
        echo "   Consider using a location with more space for large datasets"
        read -p "Continue with current location? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "💡 Alternative: Run with custom path:"
            echo "   sudo $0 /home/zabbix_anomaly_detection"
            exit 1
        fi
    else
        echo "✅ Sufficient disk space available"
    fi
}

# Main installation process
main() {
    echo "🚀 Starting RHEL installation process..."
    
    # Check disk space first
    check_disk_space
    
    # Check internet connectivity
    if ! ping -c 1 google.com >/dev/null 2>&1; then
        echo "⚠️  No internet connectivity detected"
        echo "   This installation requires internet access for package downloads"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Installation steps
    install_epel
    install_system_packages
    create_service_user
    install_python_packages
    setup_application
    create_systemd_service
    setup_log_rotation
    setup_firewall
    create_helper_scripts
    setup_selinux
    
    echo ""
    echo "🎉 Installation completed successfully!"
    echo "======================================"
    echo ""
    echo "📋 Next steps:"
    echo "1. Configure Zabbix connection:"
    echo "   sudo nano $INSTALL_DIR/config.json"
    echo ""
    echo "2. Test the connection:"
    echo "   $INSTALL_DIR/test.sh"
    echo ""
    echo "3. Collect training data:"
    echo "   sudo -u $SERVICE_USER python3 $INSTALL_DIR/get_data.py --collect-history --hours 48"
    echo ""
    echo "4. Train the initial model:"
    echo "   sudo -u $SERVICE_USER python3 $INSTALL_DIR/train_model.py --data $INSTALL_DIR/data/historical_data_*.csv"
    echo ""
    echo "5. Start the service:"
    echo "   $INSTALL_DIR/start.sh"
    echo ""
    echo "6. View the dashboard:"
    echo "   http://$(hostname -I | awk '{print $1}'):8050"
    echo ""
    echo "📝 Management commands:"
    echo "   Start:  $INSTALL_DIR/start.sh"
    echo "   Stop:   $INSTALL_DIR/stop.sh"
    echo "   Status: $INSTALL_DIR/status.sh"
    echo "   Test:   $INSTALL_DIR/test.sh"
    echo ""
    echo "📊 View logs: sudo journalctl -u $SERVICE_NAME -f"
    echo ""
    echo "🏭 System ready for industrial anomaly detection!"
}

# Run main installation
main