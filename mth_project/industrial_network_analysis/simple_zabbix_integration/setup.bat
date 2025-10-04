@echo off
REM 🏭 Quick Setup Script for Simplified Zabbix Integration (Windows)

echo 🚀 Setting up Simplified Zabbix Integration for Industrial Anomaly Detection
echo ==========================================================================

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not installed
    pause
    exit /b 1
)

echo 🐍 Python version:
python --version

REM Install Python dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully

REM Create necessary directories
echo 📁 Creating directories...
if not exist "data" mkdir data
if not exist "temp_data" mkdir temp_data
if not exist "trained_model" mkdir trained_model

echo ✅ Directories created

REM Check if config.json exists and is configured
if exist "config.json" (
    findstr "your-zabbix-server" config.json >nul
    if not errorlevel 1 (
        echo ⚠️  Configuration needed:
        echo    Please edit config.json with your Zabbix server details
        echo    - Update URL to your Zabbix server
        echo    - Set correct username and password
        echo    - Configure host groups and search criteria
    ) else (
        echo ✅ Configuration file found
    )
) else (
    echo ⚠️  No configuration file found - will be created on first run
)

echo.
echo 🎯 Next Steps:
echo ==============
echo 1. Edit config.json with your Zabbix server details
echo 2. Test connection: python get_data.py --test
echo 3. Collect training data: python get_data.py --collect-history --hours 48
echo 4. Train model: python train_model.py --data data\historical_data_*.csv
echo 5. Start monitoring: python main_forecasting.py
echo 6. View dashboard: http://localhost:8050
echo.
echo ✅ Setup completed! Ready for industrial anomaly detection.

pause