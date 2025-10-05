#!/bin/bash
# Fixed RHEL Installation Script for Industrial Network Anomaly Detection
# Resolves Zabbix dependency conflicts and ensures Python environment is ready

set -e  # Exit on any error

echo "🏭 Fixed RHEL Installation for Industrial Network Anomaly Detection"
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root for security reasons"
   log_info "Please run as regular user. Sudo will be used when needed."
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: System Update and Basic Dependencies
log_info "Step 1: Updating system and installing basic dependencies..."
sudo dnf update -y

# Install EPEL and development tools
sudo dnf install -y epel-release
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y git curl wget vim htop

log_success "System updated and basic tools installed"

# Step 2: Fix Zabbix Repository Issues
log_info "Step 2: Fixing Zabbix repository conflicts..."

# Remove conflicting Zabbix packages and repos
log_warning "Removing conflicting Zabbix packages..."
sudo dnf remove -y zabbix* || true

# Clean up old repositories
sudo rm -f /etc/yum.repos.d/zabbix*.repo

# Install specific Zabbix 6.0 LTS version (more stable for production)
log_info "Installing Zabbix 6.0 LTS repository..."
sudo rpm -Uvh https://repo.zabbix.com/zabbix/6.0/rhel/9/x86_64/zabbix-release-6.0-4.el9.noarch.rpm

# Clean cache and update
sudo dnf clean all
sudo dnf makecache

log_success "Zabbix repository configured"

# Step 3: Install Python Development Environment
log_info "Step 3: Setting up Python development environment..."

# Install Python 3.11 and development packages
sudo dnf install -y python3.11 python3.11-devel python3.11-pip
sudo dnf install -y python3-virtualenv python3-wheel

# Install system libraries needed for Python packages
sudo dnf install -y gcc gcc-c++ make
sudo dnf install -y openssl-devel libffi-devel
sudo dnf install -y postgresql-devel  # For psycopg2 if needed
sudo dnf install -y mysql-devel       # For MySQL connectivity if needed

log_success "Python development environment installed"

# Step 4: Install Zabbix Components (Optional - for monitoring only)
log_info "Step 4: Installing Zabbix components (optional)..."
read -p "Do you want to install Zabbix server components? (y/N): " install_zabbix

if [[ $install_zabbix =~ ^[Yy]$ ]]; then
    log_info "Installing Zabbix 6.0 components..."
    
    # Install PostgreSQL
    sudo dnf install -y postgresql-server postgresql-contrib
    
    # Initialize PostgreSQL if not already done
    if [ ! -f /var/lib/pgsql/data/postgresql.conf ]; then
        sudo postgresql-setup --initdb
        sudo systemctl enable postgresql
        sudo systemctl start postgresql
    fi
    
    # Install Zabbix server and web interface
    sudo dnf install -y zabbix-server-pgsql zabbix-web-pgsql zabbix-apache-conf zabbix-sql-scripts zabbix-selinux-policy
    
    log_success "Zabbix components installed"
else
    log_info "Skipping Zabbix server installation (client-only mode)"
    # Install only Zabbix agent for monitoring
    sudo dnf install -y zabbix-agent2
fi

# Step 5: Create Python Virtual Environment for the Project
log_info "Step 5: Setting up Python virtual environment..."

# Navigate to project directory
cd "$HOME"
PROJECT_DIR="industrial_network_monitor"

if [ -d "$PROJECT_DIR" ]; then
    log_warning "Project directory exists, backing up..."
    mv "$PROJECT_DIR" "${PROJECT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create virtual environment with Python 3.11
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

log_success "Virtual environment created and activated"

# Step 6: Install Python Dependencies for Anomaly Detection
log_info "Step 6: Installing Python dependencies..."

# Create requirements file for the industrial monitoring system
cat > requirements.txt << 'EOF'
# Core data processing
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0

# Zabbix integration
pyzabbix>=1.0.0

# Dashboard and visualization
plotly>=5.15.0
dash>=2.10.0

# Network and HTTP
urllib3>=1.26.0
requests>=2.28.0

# System monitoring
psutil>=5.9.0

# Optional: Machine Learning (if you want to add ML-based anomaly detection later)
scikit-learn>=1.1.0

# Optional: Advanced time series analysis
statsmodels>=0.13.0

# Development and testing
pytest>=7.0.0
black>=22.0.0
flake8>=5.0.0
EOF

# Install packages
pip install -r requirements.txt

log_success "Python dependencies installed"

# Step 7: Create Basic Project Structure
log_info "Step 7: Creating project structure..."

mkdir -p {config,logs,data,scripts,tests}

# Copy the simplified monitoring script
cat > scripts/monitor.py << 'EOF'
#!/usr/bin/env python3
"""
Industrial Network Anomaly Detection - RHEL Optimized Version
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pyzabbix import ZabbixAPI
import urllib3

# Dashboard imports (with graceful fallback)
try:
    import dash
    from dash import dcc, html, Input, Output
    import plotly.graph_objs as go
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("⚠️ Dashboard not available. Run: pip install dash plotly")

urllib3.disable_warnings()

class SimpleMonitor:
    def __init__(self, config_file="../config/config.json"):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
    def _load_config(self, config_file):
        default_config = {
            "zabbix": {
                "url": "http://localhost/zabbix",
                "user": "Admin",
                "password": "zabbix"
            },
            "monitoring": {
                "update_interval": 60,
                "dashboard_port": 8052
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                for section in default_config:
                    if section in user_config:
                        default_config[section].update(user_config[section])
                return default_config
        except FileNotFoundError:
            print(f"Config file {config_file} not found, using defaults")
            return default_config
    
    def _setup_logging(self):
        os.makedirs('../logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('../logs/monitor.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def test_connection(self):
        """Test Zabbix connection"""
        try:
            zabbix_config = self.config['zabbix']
            self.logger.info(f"Testing connection to: {zabbix_config['url']}")
            
            zapi = ZabbixAPI(zabbix_config['url'])
            zapi.login(zabbix_config['user'], zabbix_config['password'])
            
            version = zapi.apiinfo.version()
            self.logger.info(f"✅ Connected to Zabbix {version}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Connection failed: {e}")
            return False
    
    def run_basic_monitor(self):
        """Run basic monitoring loop"""
        self.logger.info("🏭 Starting basic industrial network monitor...")
        
        if not self.test_connection():
            self.logger.error("Cannot connect to Zabbix. Please check configuration.")
            return
        
        # Basic monitoring loop
        cycle = 0
        while True:
            try:
                cycle += 1
                self.logger.info(f"🔄 Monitoring cycle #{cycle}")
                
                # Your monitoring logic here
                time.sleep(self.config['monitoring']['update_interval'])
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Monitoring stopped")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in cycle {cycle}: {e}")
                time.sleep(30)

if __name__ == "__main__":
    monitor = SimpleMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        monitor.test_connection()
    else:
        monitor.run_basic_monitor()
EOF

chmod +x scripts/monitor.py

# Create default configuration
cat > config/config.json << 'EOF'
{
  "zabbix": {
    "url": "http://localhost/zabbix",
    "user": "Admin",
    "password": "zabbix"
  },
  "monitoring": {
    "update_interval": 60,
    "dashboard_port": 8052,
    "search_criteria": ["cpu", "memory", "network", "temperature"],
    "host_groups": ["Linux servers"]
  },
  "logging": {
    "level": "INFO",
    "file": "../logs/monitor.log"
  }
}
EOF

log_success "Project structure created"

# Step 8: Create startup scripts
log_info "Step 8: Creating startup scripts..."

# Create activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activate the virtual environment for industrial monitoring

echo "🏭 Activating Industrial Network Monitoring Environment"
echo "======================================================"

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated"
echo "📁 Project directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo ""
echo "Available commands:"
echo "  python scripts/monitor.py --test    # Test Zabbix connection"
echo "  python scripts/monitor.py           # Run monitoring"
echo ""
echo "To deactivate: deactivate"
EOF

chmod +x activate_env.sh

# Create systemd service file (optional)
cat > industrial-monitor.service << 'EOF'
[Unit]
Description=Industrial Network Anomaly Detection Monitor
After=network.target

[Service]
Type=simple
User=monitoring
WorkingDirectory=/home/monitoring/industrial_network_monitor
Environment=PATH=/home/monitoring/industrial_network_monitor/venv/bin
ExecStart=/home/monitoring/industrial_network_monitor/venv/bin/python scripts/monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

log_success "Startup scripts created"

# Step 9: Final Setup and Testing
log_info "Step 9: Final setup and testing..."

# Test Python environment
log_info "Testing Python environment..."
python -c "import pandas, numpy, pyzabbix; print('✅ Core dependencies working')"

if [ $? -eq 0 ]; then
    log_success "Python environment is working correctly"
else
    log_error "Python environment has issues"
fi

# Create README
cat > README.md << 'EOF'
# Industrial Network Anomaly Detection - RHEL Installation

## Quick Start

1. Activate environment:
   ```bash
   ./activate_env.sh
   ```

2. Configure Zabbix connection:
   ```bash
   vim config/config.json
   ```

3. Test connection:
   ```bash
   python scripts/monitor.py --test
   ```

4. Run monitoring:
   ```bash
   python scripts/monitor.py
   ```

## Project Structure

- `config/` - Configuration files
- `scripts/` - Main monitoring scripts
- `logs/` - Log files
- `data/` - Data storage
- `venv/` - Python virtual environment

## Troubleshooting

- Check logs in `logs/monitor.log`
- Ensure Zabbix is accessible
- Verify virtual environment is activated

## Dependencies

All Python dependencies are installed in the virtual environment.
Main packages: pandas, numpy, pyzabbix, dash, plotly
EOF

log_success "Installation completed!"

echo ""
echo "🎉 INSTALLATION SUMMARY"
echo "======================"
echo "✅ System updated and development tools installed"
echo "✅ Zabbix repository conflicts resolved"
echo "✅ Python 3.11 development environment ready"
echo "✅ Virtual environment created with dependencies"
echo "✅ Project structure created"
echo "✅ Startup scripts ready"
echo ""
echo "📁 Project location: $HOME/$PROJECT_DIR"
echo "🐍 Virtual environment: $HOME/$PROJECT_DIR/venv"
echo ""
echo "🚀 NEXT STEPS:"
echo "1. cd $HOME/$PROJECT_DIR"
echo "2. ./activate_env.sh"
echo "3. Edit config/config.json with your Zabbix details"
echo "4. python scripts/monitor.py --test"
echo ""
echo "🔧 To copy your existing monitoring code:"
echo "   Copy your main_forecasting_fixed.py to the scripts/ directory"
echo ""

# Deactivate virtual environment
deactivate

log_success "Setup complete! Virtual environment deactivated."
log_info "Run './activate_env.sh' in the project directory to start working."