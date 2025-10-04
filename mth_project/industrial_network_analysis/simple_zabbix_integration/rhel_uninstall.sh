#!/bin/bash

# 🏭 RHEL Uninstaller for Zabbix Anomaly Detection

set -e

SERVICE_NAME="zabbix-anomaly-detection"
INSTALL_DIR="/opt/zabbix_anomaly_detection"
SERVICE_USER="anomaly"

echo "🗑️  RHEL Uninstaller - Zabbix Industrial Anomaly Detection"
echo "========================================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run as root (use sudo)"
    exit 1
fi

# Confirmation prompt
echo "⚠️  This will completely remove the Zabbix Anomaly Detection system"
echo "   Installation directory: $INSTALL_DIR"
echo "   Service: $SERVICE_NAME"
echo "   User: $SERVICE_USER"
echo ""
read -p "Are you sure you want to continue? (type 'yes' to confirm): " -r
if [ "$REPLY" != "yes" ]; then
    echo "❌ Uninstallation cancelled"
    exit 1
fi

echo "🛑 Stopping and removing service..."

# Stop and disable service
if systemctl is-active --quiet $SERVICE_NAME; then
    systemctl stop $SERVICE_NAME
    echo "✅ Service stopped"
fi

if systemctl is-enabled --quiet $SERVICE_NAME; then
    systemctl disable $SERVICE_NAME
    echo "✅ Service disabled"
fi

# Remove systemd service file
if [ -f "/etc/systemd/system/${SERVICE_NAME}.service" ]; then
    rm -f "/etc/systemd/system/${SERVICE_NAME}.service"
    systemctl daemon-reload
    echo "✅ Service file removed"
fi

# Remove logrotate configuration
if [ -f "/etc/logrotate.d/$SERVICE_NAME" ]; then
    rm -f "/etc/logrotate.d/$SERVICE_NAME"
    echo "✅ Log rotation configuration removed"
fi

# Remove firewall rule
if systemctl is-active --quiet firewalld; then
    echo "🔥 Removing firewall rules..."
    firewall-cmd --permanent --remove-port=8050/tcp 2>/dev/null || true
    firewall-cmd --reload 2>/dev/null || true
    echo "✅ Firewall rules removed"
fi

# Remove SELinux configuration
if command -v getenforce >/dev/null 2>&1 && [ "$(getenforce)" != "Disabled" ]; then
    echo "🔒 Removing SELinux configuration..."
    semanage port -d -t http_port_t -p tcp 8050 2>/dev/null || true
    semanage fcontext -d "$INSTALL_DIR/.*\.py" 2>/dev/null || true
    echo "✅ SELinux configuration removed"
fi

# Remove installation directory
if [ -d "$INSTALL_DIR" ]; then
    echo "📁 Removing installation directory..."
    rm -rf "$INSTALL_DIR"
    echo "✅ Installation directory removed"
fi

# Remove service user
if id "$SERVICE_USER" &>/dev/null; then
    echo "👤 Removing service user..."
    userdel "$SERVICE_USER" 2>/dev/null || true
    echo "✅ Service user removed"
fi

# Optional: Remove Python packages (ask user)
echo ""
read -p "Remove Python packages installed for this system? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🐍 Removing Python packages..."
    python3 -m pip uninstall -y pyzabbix tensorflow pandas numpy scikit-learn plotly dash 2>/dev/null || true
    echo "✅ Python packages removed"
fi

# Optional: Remove system packages (ask user)
echo ""
read -p "Remove system development packages? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Removing system packages..."
    
    # Detect RHEL version
    if [ -f /etc/redhat-release ]; then
        RHEL_VERSION=$(cat /etc/redhat-release | grep -oE '[0-9]+' | head -1)
    else
        RHEL_VERSION=8
    fi
    
    case $RHEL_VERSION in
        7)
            yum remove -y python3-devel gcc gcc-c++ make openssl-devel libffi-devel 2>/dev/null || true
            ;;
        8|9)
            dnf remove -y python3-devel gcc gcc-c++ make openssl-devel libffi-devel 2>/dev/null || true
            ;;
        *)
            dnf remove -y python3-devel gcc gcc-c++ make openssl-devel libffi-devel 2>/dev/null || \
            yum remove -y python3-devel gcc gcc-c++ make openssl-devel libffi-devel 2>/dev/null || true
            ;;
    esac
    
    echo "✅ System packages removed"
fi

echo ""
echo "✅ Uninstallation completed successfully!"
echo "======================================="
echo ""
echo "🧹 The following have been removed:"
echo "   - Systemd service: $SERVICE_NAME"
echo "   - Installation directory: $INSTALL_DIR"
echo "   - Service user: $SERVICE_USER"
echo "   - Firewall rules for port 8050"
echo "   - Log rotation configuration"
echo "   - SELinux policies"
echo ""
echo "💾 Note: System packages and Python packages may have been retained"
echo "   if you chose not to remove them."
echo ""
echo "🏭 Zabbix Anomaly Detection system has been completely removed."