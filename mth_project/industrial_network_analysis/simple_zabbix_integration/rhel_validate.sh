#!/bin/bash

# 🧪 RHEL System Validation for Zabbix Anomaly Detection

echo "🧪 RHEL System Validation - Zabbix Anomaly Detection"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
WARN=0

# Function to print test results
print_result() {
    local test_name="$1"
    local result="$2"
    local message="$3"
    
    case $result in
        "PASS")
            echo -e "✅ ${GREEN}PASS${NC}: $test_name"
            ((PASS++))
            ;;
        "FAIL")
            echo -e "❌ ${RED}FAIL${NC}: $test_name - $message"
            ((FAIL++))
            ;;
        "WARN")
            echo -e "⚠️  ${YELLOW}WARN${NC}: $test_name - $message"
            ((WARN++))
            ;;
        "INFO")
            echo -e "ℹ️  ${BLUE}INFO${NC}: $test_name - $message"
            ;;
    esac
}

# System Information
echo ""
echo "🖥️  System Information"
echo "===================="

if [ -f /etc/redhat-release ]; then
    OS_INFO=$(cat /etc/redhat-release)
    print_result "Operating System" "INFO" "$OS_INFO"
    
    RHEL_VERSION=$(echo "$OS_INFO" | grep -oE '[0-9]+' | head -1)
    if [ "$RHEL_VERSION" -ge 7 ] && [ "$RHEL_VERSION" -le 9 ]; then
        print_result "RHEL Version Support" "PASS"
    else
        print_result "RHEL Version Support" "WARN" "Version $RHEL_VERSION may not be fully tested"
    fi
else
    print_result "Operating System" "FAIL" "Not a Red Hat based system"
fi

# Architecture check
ARCH=$(uname -m)
print_result "Architecture" "INFO" "$ARCH"
if [ "$ARCH" = "x86_64" ]; then
    print_result "Architecture Support" "PASS"
else
    print_result "Architecture Support" "WARN" "Architecture $ARCH may have limited package availability"
fi

# Hardware Requirements
echo ""
echo "🔧 Hardware Requirements"
echo "======================="

# CPU cores
CPU_CORES=$(nproc)
print_result "CPU Cores" "INFO" "$CPU_CORES cores"
if [ "$CPU_CORES" -ge 2 ]; then
    print_result "CPU Requirements" "PASS"
else
    print_result "CPU Requirements" "WARN" "Minimum 2 cores recommended, found $CPU_CORES"
fi

# Memory
MEMORY_GB=$(free -g | awk 'NR==2{print $2}')
print_result "Memory" "INFO" "${MEMORY_GB}GB"
if [ "$MEMORY_GB" -ge 4 ]; then
    print_result "Memory Requirements" "PASS"
elif [ "$MEMORY_GB" -ge 2 ]; then
    print_result "Memory Requirements" "WARN" "4GB recommended, found ${MEMORY_GB}GB"
else
    print_result "Memory Requirements" "FAIL" "Minimum 2GB required, found ${MEMORY_GB}GB"
fi

# Disk space requirements
echo "💾 Checking disk space requirements..."

# Check /opt directory (main installation)
OPT_AVAIL_GB=$(df /opt 2>/dev/null | awk 'NR==2 {print int($4/1024/1024)}' || df / | awk 'NR==2 {print int($4/1024/1024)}')
OPT_MOUNT=$(df /opt 2>/dev/null | awk 'NR==2 {print $6}' || echo "/")
print_result "/opt Directory Space" "INFO" "${OPT_AVAIL_GB}GB available (mounted on $OPT_MOUNT)"

if [ "$OPT_AVAIL_GB" -ge 5 ]; then
    print_result "/opt Space Requirements" "PASS"
elif [ "$OPT_AVAIL_GB" -ge 2 ]; then
    print_result "/opt Space Requirements" "WARN" "5GB recommended for models and data, found ${OPT_AVAIL_GB}GB"
else
    print_result "/opt Space Requirements" "FAIL" "Minimum 2GB required for installation, found ${OPT_AVAIL_GB}GB"
fi

# Check /var directory (logs and temporary data)
VAR_AVAIL_GB=$(df /var 2>/dev/null | awk 'NR==2 {print int($4/1024/1024)}' || df / | awk 'NR==2 {print int($4/1024/1024)}')
VAR_MOUNT=$(df /var 2>/dev/null | awk 'NR==2 {print $6}' || echo "/")
print_result "/var Directory Space" "INFO" "${VAR_AVAIL_GB}GB available (mounted on $VAR_MOUNT)"

if [ "$VAR_AVAIL_GB" -ge 2 ]; then
    print_result "/var Space Requirements" "PASS"
elif [ "$VAR_AVAIL_GB" -ge 1 ]; then
    print_result "/var Space Requirements" "WARN" "2GB recommended for logs, found ${VAR_AVAIL_GB}GB"
else
    print_result "/var Space Requirements" "FAIL" "Minimum 1GB required for logs, found ${VAR_AVAIL_GB}GB"
fi

# Check root filesystem (general system space)
ROOT_AVAIL_GB=$(df / | awk 'NR==2 {print int($4/1024/1024)}')
print_result "Root Filesystem Space" "INFO" "${ROOT_AVAIL_GB}GB available"

if [ "$ROOT_AVAIL_GB" -ge 2 ]; then
    print_result "Root Space Requirements" "PASS"
elif [ "$ROOT_AVAIL_GB" -ge 1 ]; then
    print_result "Root Space Requirements" "WARN" "Root filesystem is getting full (${ROOT_AVAIL_GB}GB available)"
else
    print_result "Root Space Requirements" "FAIL" "Root filesystem critically low (${ROOT_AVAIL_GB}GB available)"
fi

# Overall disk space assessment
TOTAL_NEEDED=7  # 5GB for /opt + 2GB for /var
TOTAL_AVAILABLE=$((OPT_AVAIL_GB + VAR_AVAIL_GB))

print_result "Total Disk Assessment" "INFO" "Need ~${TOTAL_NEEDED}GB total, available across filesystems: ${TOTAL_AVAILABLE}GB"

if [ "$TOTAL_AVAILABLE" -ge "$TOTAL_NEEDED" ]; then
    print_result "Overall Disk Requirements" "PASS"
else
    print_result "Overall Disk Requirements" "WARN" "Tight disk space - monitor usage during operation"
fi

# Network Connectivity
echo ""
echo "🌐 Network Connectivity"
echo "======================"

# Internet connectivity
if ping -c 1 google.com >/dev/null 2>&1; then
    print_result "Internet Connectivity" "PASS"
else
    print_result "Internet Connectivity" "FAIL" "Required for package installation"
fi

# DNS resolution
if nslookup google.com >/dev/null 2>&1; then
    print_result "DNS Resolution" "PASS"
else
    print_result "DNS Resolution" "WARN" "DNS issues may affect package downloads"
fi

# Package Manager
echo ""
echo "📦 Package Manager"
echo "=================="

# Check for yum/dnf
if command -v dnf >/dev/null 2>&1; then
    print_result "Package Manager" "INFO" "dnf available"
    print_result "Package Manager Support" "PASS"
elif command -v yum >/dev/null 2>&1; then
    print_result "Package Manager" "INFO" "yum available"
    print_result "Package Manager Support" "PASS"
else
    print_result "Package Manager Support" "FAIL" "Neither yum nor dnf found"
fi

# EPEL repository
if [ "$RHEL_VERSION" ]; then
    case $RHEL_VERSION in
        7)
            if yum repolist | grep -q epel; then
                print_result "EPEL Repository" "PASS"
            else
                print_result "EPEL Repository" "WARN" "EPEL not configured (will be installed)"
            fi
            ;;
        8|9)
            if dnf repolist | grep -q epel; then
                print_result "EPEL Repository" "PASS"
            else
                print_result "EPEL Repository" "WARN" "EPEL not configured (will be installed)"
            fi
            ;;
    esac
fi

# Python Environment  
echo ""
echo "🐍 Python Environment"
echo "===================="

# Python 3
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_result "Python 3" "INFO" "Version $PYTHON_VERSION"
    
    # Check Python version (3.7+)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 7 ]; then
        print_result "Python Version" "PASS"
    else
        print_result "Python Version" "WARN" "Python 3.7+ recommended, found $PYTHON_VERSION"
    fi
else
    print_result "Python 3" "FAIL" "Python 3 not found"
fi

# pip
if command -v pip3 >/dev/null 2>&1; then
    PIP_VERSION=$(pip3 --version | awk '{print $2}')
    print_result "pip3" "PASS" "Version $PIP_VERSION"
else
    print_result "pip3" "WARN" "pip3 not found (will be installed)"
fi

# Development Tools
echo ""
echo "🔨 Development Environment"
echo "========================="

# GCC
if command -v gcc >/dev/null 2>&1; then
    GCC_VERSION=$(gcc --version | head -n1 | awk '{print $3}')
    print_result "GCC" "PASS" "Version $GCC_VERSION"
else
    print_result "GCC" "WARN" "GCC not found (will be installed)"
fi

# Make  
if command -v make >/dev/null 2>&1; then
    MAKE_VERSION=$(make --version | head -n1 | awk '{print $3}')
    print_result "Make" "PASS" "Version $MAKE_VERSION"
else
    print_result "Make" "WARN" "Make not found (will be installed)"
fi

# Security Configuration
echo ""
echo "🔒 Security Configuration"
echo "========================"

# SELinux
if command -v getenforce >/dev/null 2>&1; then
    SELINUX_STATUS=$(getenforce)
    print_result "SELinux" "INFO" "$SELINUX_STATUS"
    
    if [ "$SELINUX_STATUS" = "Enforcing" ]; then
        print_result "SELinux Support" "PASS" "Will configure policies"
    elif [ "$SELINUX_STATUS" = "Permissive" ]; then
        print_result "SELinux Support" "WARN" "Permissive mode detected"
    else
        print_result "SELinux Support" "INFO" "Disabled"
    fi
else
    print_result "SELinux" "INFO" "Not available"
fi

# Firewalld
if systemctl is-active --quiet firewalld; then
    print_result "Firewalld" "INFO" "Active (will configure rules)"
    print_result "Firewalld Support" "PASS"
elif command -v firewall-cmd >/dev/null 2>&1; then
    print_result "Firewalld" "INFO" "Inactive"
    print_result "Firewalld Support" "WARN" "Firewall not running"
else
    print_result "Firewalld" "INFO" "Not installed"
fi

# Systemd
if command -v systemctl >/dev/null 2>&1; then
    print_result "Systemd" "PASS"
else
    print_result "Systemd" "FAIL" "Required for service management"
fi

# User Privileges
echo ""
echo "👤 User Privileges"
echo "=================="

# Root/sudo access
if [ "$EUID" -eq 0 ]; then
    print_result "Root Access" "PASS" "Running as root"
elif sudo -n true 2>/dev/null; then
    print_result "Sudo Access" "PASS" "Passwordless sudo available"
else
    print_result "Admin Access" "WARN" "Root/sudo access required for installation"
fi

# Port Availability
echo ""
echo "🔌 Port Availability"
echo "==================="

# Check if port 8050 is available
if netstat -tln 2>/dev/null | grep -q ":8050 "; then
    print_result "Port 8050" "WARN" "Port already in use"
else
    print_result "Port 8050" "PASS" "Available for dashboard"
fi

# Summary
echo ""
echo "📊 Validation Summary"
echo "===================="
echo -e "✅ ${GREEN}PASSED${NC}: $PASS tests"
echo -e "⚠️  ${YELLOW}WARNINGS${NC}: $WARN tests"
echo -e "❌ ${RED}FAILED${NC}: $FAIL tests"

echo ""
if [ "$FAIL" -eq 0 ]; then
    if [ "$WARN" -eq 0 ]; then
        echo -e "🎉 ${GREEN}System is ready for installation!${NC}"
        echo "   You can proceed with: sudo ./rhel_install.sh"
    else
        echo -e "⚠️  ${YELLOW}System is mostly ready for installation${NC}"
        echo "   Warnings noted above should be reviewed"
        echo "   You can proceed with: sudo ./rhel_install.sh"
    fi
else
    echo -e "❌ ${RED}System requires attention before installation${NC}"
    echo "   Please resolve the failed tests above"
fi

echo ""
echo "📋 Next Steps:"
echo "=============="
echo "1. Address any failed requirements above"
echo "2. Review warnings and plan accordingly"  
echo "3. Run installation: sudo ./rhel_install.sh"
echo "4. Configure Zabbix connection in config.json"
echo "5. Test system: /opt/zabbix_anomaly_detection/test.sh"

exit $FAIL