#!/bin/bash

# 🏭 Quick Setup Script for Simplified Zabbix Integration

echo "🚀 Setting up Simplified Zabbix Integration for Industrial Anomaly Detection"
echo "=========================================================================="

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "🐍 Python version: $(python3 --version)"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p temp_data
mkdir -p trained_model

echo "✅ Directories created"

# Check if config.json exists and is configured
if [ -f "config.json" ]; then
    if grep -q "your-zabbix-server" config.json; then
        echo "⚠️  Configuration needed:"
        echo "   Please edit config.json with your Zabbix server details"
        echo "   - Update URL to your Zabbix server"
        echo "   - Set correct username and password"
        echo "   - Configure host groups and search criteria"
    else
        echo "✅ Configuration file found"
    fi
else
    echo "⚠️  No configuration file found - will be created on first run"
fi

echo ""
echo "🎯 Next Steps:"
echo "=============="
echo "1. Edit config.json with your Zabbix server details"
echo "2. Test connection: python3 get_data.py --test"
echo "3. Collect training data: python3 get_data.py --collect-history --hours 48"
echo "4. Train model: python3 train_model.py --data data/historical_data_*.csv"
echo "5. Start monitoring: python3 main_forecasting.py"
echo "6. View dashboard: http://localhost:8050"
echo ""
echo "✅ Setup completed! Ready for industrial anomaly detection."