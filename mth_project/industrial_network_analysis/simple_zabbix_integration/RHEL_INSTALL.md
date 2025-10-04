# 🏭 RHEL Installation Guide

## System Requirements

### Supported RHEL Versions
- Red Hat Enterprise Linux 7.x
- Red Hat Enterprise Linux 8.x  
- Red Hat Enterprise Linux 9.x
- CentOS 7.x/8.x/9.x
- Rocky Linux 8.x/9.x
- AlmaLinux 8.x/9.x

### Minimum Hardware Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 10GB free space
- **Network**: Internet access for installation, Zabbix server access for operation

### Required Privileges
- Root access (sudo) for installation
- Network access to Zabbix server
- Firewall configuration rights (if applicable)

## Quick Installation

### 1. Download and Run Installer

```bash
# Make installer executable
chmod +x rhel_install.sh

# Run installation (requires root)
sudo ./rhel_install.sh
```

### 2. Configure Zabbix Connection

```bash
# Edit configuration
sudo nano /opt/zabbix_anomaly_detection/config.json
```

Update these settings:
```json
{
  "zabbix": {
    "url": "http://your-zabbix-server/zabbix",
    "user": "Admin",
    "password": "your-password"
  }
}
```

### 3. Test Installation

```bash
# Test system components
/opt/zabbix_anomaly_detection/test.sh
```

### 4. Start Monitoring

```bash
# Start the service
/opt/zabbix_anomaly_detection/start.sh

# View dashboard
# http://your-server-ip:8050
```

## Detailed Installation Process

### What the Installer Does

1. **System Packages**:
   - Installs EPEL repository
   - Installs Python 3 and development tools
   - Installs required system libraries

2. **Python Environment**:
   - Upgrades pip to latest version
   - Installs TensorFlow with RHEL-compatible versions
   - Installs all required dependencies

3. **Service Setup**:
   - Creates dedicated service user (`anomaly`)
   - Sets up systemd service for automatic startup
   - Configures proper permissions and security

4. **System Integration**:
   - Configures firewall rules (port 8050)
   - Sets up log rotation
   - Configures SELinux policies (if enabled)
   - Creates management scripts

### Directory Structure

```
/opt/zabbix_anomaly_detection/
├── config.json              # Configuration file
├── get_data.py              # Data collection script
├── train_model.py           # Model training script
├── main_forecasting.py      # Main monitoring service
├── classification_wrapper.py # Classification integration
├── requirements.txt         # Python dependencies
├── README.md               # Documentation
├── start.sh                # Service start script
├── stop.sh                 # Service stop script
├── status.sh               # Service status script
├── test.sh                 # System test script
├── data/                   # Training data storage
├── temp_data/              # Temporary cycle data
├── trained_model/          # Saved models
└── logs/                   # Application logs
```

## Service Management

### Systemd Service

The installer creates a systemd service: `zabbix-anomaly-detection`

```bash
# Start service
sudo systemctl start zabbix-anomaly-detection

# Stop service  
sudo systemctl stop zabbix-anomaly-detection

# Restart service
sudo systemctl restart zabbix-anomaly-detection

# Check status
sudo systemctl status zabbix-anomaly-detection

# Enable auto-start on boot
sudo systemctl enable zabbix-anomaly-detection

# View logs
sudo journalctl -u zabbix-anomaly-detection -f
```

### Helper Scripts

```bash
# Quick start/stop
/opt/zabbix_anomaly_detection/start.sh
/opt/zabbix_anomaly_detection/stop.sh

# Check system status
/opt/zabbix_anomaly_detection/status.sh

# Test system components
/opt/zabbix_anomaly_detection/test.sh
```

## Security Configuration

### Firewall

The installer automatically configures firewall rules:

```bash
# Manual firewall configuration if needed
sudo firewall-cmd --permanent --add-port=8050/tcp
sudo firewall-cmd --reload
```

### SELinux

If SELinux is enabled, the installer configures:

```bash
# Allow network connections
sudo setsebool -P httpd_can_network_connect 1

# Configure port access
sudo semanage port -a -t http_port_t -p tcp 8050
```

### Service User

The service runs as a dedicated user `anomaly` with minimal privileges:

```bash
# Check service user
id anomaly

# Run commands as service user
sudo -u anomaly python3 /opt/zabbix_anomaly_detection/get_data.py --test
```

## Troubleshooting

### Installation Issues

**EPEL Repository Error**:
```bash
# Manual EPEL installation
# RHEL 8/9
sudo dnf install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm

# RHEL 7
sudo yum install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
```

**Python Package Installation Fails**:
```bash
# Upgrade pip and setuptools
sudo python3 -m pip install --upgrade pip setuptools wheel

# Install with no cache
sudo python3 -m pip install --no-cache-dir -r requirements.txt
```

**TensorFlow Installation Issues**:
```bash
# Install specific TensorFlow version for RHEL
sudo python3 -m pip install tensorflow==2.12.0
```

### Runtime Issues

**Service Won't Start**:
```bash
# Check detailed logs
sudo journalctl -u zabbix-anomaly-detection -f

# Check configuration
sudo -u anomaly python3 /opt/zabbix_anomaly_detection/main_forecasting.py --test
```

**Dashboard Not Accessible**:
```bash
# Check if service is running
sudo netstat -tlnp | grep 8050

# Check firewall
sudo firewall-cmd --list-ports

# Check SELinux logs
sudo ausearch -m avc -ts recent
```

**Zabbix Connection Issues**:
```bash
# Test Zabbix connection
sudo -u anomaly python3 /opt/zabbix_anomaly_detection/get_data.py --test

# Check network connectivity
telnet your-zabbix-server 80
```

### Performance Tuning

**Memory Usage**:
```bash
# Monitor memory usage
ps aux | grep python3

# Adjust service limits in systemd
sudo systemctl edit zabbix-anomaly-detection
```

Add to override file:
```ini
[Service]
MemoryMax=2G
MemoryHigh=1.5G
```

**CPU Usage**:
```bash
# Monitor CPU usage
top -p $(pgrep -f main_forecasting.py)

# Set CPU limits
sudo systemctl edit zabbix-anomaly-detection
```

Add to override file:
```ini
[Service]
CPUQuota=200%
```

## Monitoring and Maintenance

### Log Management

Logs are automatically rotated daily and kept for 30 days:

```bash
# View current logs
sudo journalctl -u zabbix-anomaly-detection -f

# View specific time range
sudo journalctl -u zabbix-anomaly-detection --since "1 hour ago"

# View application logs
sudo tail -f /opt/zabbix_anomaly_detection/logs/*.log
```

### Health Checks

```bash
# System health check
/opt/zabbix_anomaly_detection/status.sh

# Test all components
/opt/zabbix_anomaly_detection/test.sh

# Check model status
sudo -u anomaly ls -la /opt/zabbix_anomaly_detection/trained_model/
```

### Updates

```bash
# Update Python packages
sudo python3 -m pip install --upgrade -r /opt/zabbix_anomaly_detection/requirements.txt

# Restart service after updates
sudo systemctl restart zabbix-anomaly-detection
```

## Production Recommendations

### System Monitoring

Set up monitoring for:
- Service status (`systemctl status zabbix-anomaly-detection`)
- Dashboard accessibility (port 8050)
- Log file growth
- Memory and CPU usage
- Disk space in `/opt/zabbix_anomaly_detection/`

### Backup Strategy

Important files to backup:
- `/opt/zabbix_anomaly_detection/config.json`
- `/opt/zabbix_anomaly_detection/trained_model/`
- `/opt/zabbix_anomaly_detection/data/` (training data)

### Regular Maintenance

1. **Weekly**: Check service status and logs
2. **Monthly**: Review model performance metrics
3. **Quarterly**: Update Python packages and retrain models
4. **Annually**: Review and update system packages

## Support

For issues specific to RHEL installation:

1. Check system logs: `sudo journalctl -u zabbix-anomaly-detection`
2. Verify network connectivity to Zabbix server
3. Ensure all required packages are installed
4. Check firewall and SELinux configurations
5. Verify service user permissions

The system is designed to be robust and self-healing, with automatic restarts and comprehensive error handling for production environments.