#!/bin/bash
# Zabbix Cleanup and Reinstall Script for RHEL
# Resolves package dependency conflicts

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

echo "🧹 Zabbix Cleanup and Reinstall Script"
echo "======================================"

# Check if running as root or with sudo
if [[ $EUID -ne 0 ]]; then
   log_error "This script must be run as root or with sudo"
   exit 1
fi

# Step 1: Complete Zabbix removal
log_info "Step 1: Removing all Zabbix packages and configurations..."

# Stop all Zabbix services
systemctl stop zabbix-server zabbix-agent zabbix-agent2 httpd postgresql || true

# Remove all Zabbix packages
dnf remove -y zabbix* || true

# Remove repository files
rm -f /etc/yum.repos.d/zabbix*.repo

# Clean package cache
dnf clean all
dnf makecache

log_success "Zabbix packages removed"

# Step 2: Fix SELinux policy issues
log_info "Step 2: Fixing SELinux policy conflicts..."

# Update SELinux policies first
dnf update -y selinux-policy selinux-policy-targeted

# If SELinux is causing issues, you can temporarily set it to permissive
# setenforce 0  # Uncomment if needed

log_success "SELinux policies updated"

# Step 3: Choose Zabbix version
log_info "Step 3: Installing Zabbix (choosing stable version)..."

echo "Which Zabbix version would you like to install?"
echo "1) Zabbix 6.0 LTS (Recommended - Long Term Support)"
echo "2) Zabbix 7.0 (Latest - may have compatibility issues)"
echo "3) Skip Zabbix installation (Python monitoring only)"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        ZABBIX_VERSION="6.0"
        ZABBIX_REPO="https://repo.zabbix.com/zabbix/6.0/rhel/9/x86_64/zabbix-release-6.0-4.el9.noarch.rpm"
        ;;
    2)
        ZABBIX_VERSION="7.0"
        ZABBIX_REPO="https://repo.zabbix.com/zabbix/7.0/rhel/9/x86_64/zabbix-release-7.0-2.el9.noarch.rpm"
        ;;
    3)
        log_info "Skipping Zabbix installation - Python monitoring only"
        SKIP_ZABBIX=true
        ;;
    *)
        log_error "Invalid choice. Defaulting to Zabbix 6.0 LTS"
        ZABBIX_VERSION="6.0"
        ZABBIX_REPO="https://repo.zabbix.com/zabbix/6.0/rhel/9/x86_64/zabbix-release-6.0-4.el9.noarch.rpm"
        ;;
esac

if [ "$SKIP_ZABBIX" != "true" ]; then
    # Install Zabbix repository
    log_info "Installing Zabbix $ZABBIX_VERSION repository..."
    rpm -Uvh $ZABBIX_REPO

    # Update package cache
    dnf clean all
    dnf makecache

    # Install PostgreSQL (recommended database)
    log_info "Installing PostgreSQL..."
    dnf install -y postgresql-server postgresql-contrib

    # Initialize PostgreSQL if needed
    if [ ! -f /var/lib/pgsql/data/postgresql.conf ]; then
        postgresql-setup --initdb
        systemctl enable postgresql
        systemctl start postgresql
    fi

    # Install Zabbix server components
    log_info "Installing Zabbix $ZABBIX_VERSION server components..."
    dnf install -y zabbix-server-pgsql zabbix-web-pgsql zabbix-apache-conf zabbix-sql-scripts

    # Install Zabbix agent
    if [ "$ZABBIX_VERSION" = "6.0" ]; then
        dnf install -y zabbix-agent
    else
        dnf install -y zabbix-agent2
    fi

    log_success "Zabbix $ZABBIX_VERSION installed successfully"

    # Step 4: Basic Zabbix configuration
    log_info "Step 4: Basic Zabbix configuration..."

    # Create Zabbix database
    log_info "Setting up Zabbix database..."
    
    # Create database and user
    sudo -u postgres createuser --pwprompt zabbix || true
    sudo -u postgres createdb -O zabbix zabbix || true

    # Import initial schema
    if [ "$ZABBIX_VERSION" = "6.0" ]; then
        zcat /usr/share/doc/zabbix-sql-scripts/postgresql/server.sql.gz | sudo -u zabbix psql zabbix || true
    else
        zcat /usr/share/zabbix-sql-scripts/postgresql/server.sql.gz | sudo -u zabbix psql zabbix || true
    fi

    # Configure Zabbix server
    log_info "Configuring Zabbix server..."
    sed -i 's/# DBPassword=/DBPassword=zabbix_password/' /etc/zabbix/zabbix_server.conf

    # Configure PHP for Zabbix web interface
    sed -i 's/; date.timezone =/date.timezone = America\/New_York/' /etc/php.ini

    # Start services
    systemctl restart zabbix-server zabbix-agent httpd postgresql
    systemctl enable zabbix-server zabbix-agent httpd postgresql

    log_success "Zabbix configuration completed"

    echo ""
    echo "🎉 ZABBIX INSTALLATION COMPLETED"
    echo "================================"
    echo "✅ Zabbix $ZABBIX_VERSION installed and configured"
    echo "🌐 Web interface: http://$(hostname -I | awk '{print $1}')/zabbix"
    echo "👤 Default login: Admin / zabbix"
    echo ""
    echo "📋 Next steps:"
    echo "1. Access the web interface and complete setup"
    echo "2. Change default password"
    echo "3. Configure monitoring hosts"
    echo ""
else
    log_info "Zabbix installation skipped - ready for Python monitoring only"
fi

# Step 5: Verify Python environment for monitoring
log_info "Step 5: Verifying Python environment..."

# Check if Python 3.11 is available
if command -v python3.11 >/dev/null 2>&1; then
    log_success "Python 3.11 is available"
else
    log_info "Installing Python 3.11..."
    dnf install -y python3.11 python3.11-devel python3.11-pip
fi

# Install system packages needed for Python packages
dnf install -y gcc gcc-c++ make openssl-devel libffi-devel

log_success "Python environment ready"

echo ""
echo "🏭 INDUSTRIAL NETWORK MONITORING SETUP"
echo "======================================"
echo "✅ System is ready for industrial network anomaly detection"
echo "🐍 Python environment configured"
if [ "$SKIP_ZABBIX" != "true" ]; then
    echo "📊 Zabbix $ZABBIX_VERSION installed and running"
fi
echo ""
echo "🚀 NEXT STEPS:"
echo "1. Run the dependency checker: python3.11 check_dependencies.py"
echo "2. Set up your monitoring project with the fixed Python script"
echo "3. Configure Zabbix connection details"
echo ""

log_success "Cleanup and reinstallation completed!"