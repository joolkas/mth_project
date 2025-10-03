#!/usr/bin/env python3
"""
🤖 Complete Auto-Training for Zabbix Data

Collects data from Zabbix AND trains the model automatically.
No manual steps required!
"""

import sys
import os
import json
import subprocess

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_training import collect_training_data


def auto_train_model(config_path="config.json"):
    """Complete automatic training pipeline"""
    
    print("🤖 Starting Complete Auto-Training Pipeline...")
    print("=" * 50)
    
    # Step 1: Collect training data
    print("📊 Step 1: Collecting training data from Zabbix...")
    df_forecasting, df_classification = collect_training_data(config_path, days=7)
    
    if df_forecasting is None:
        print("❌ Training data collection failed!")
        return False
    
    print(f"✅ Data collected: {df_forecasting.shape}")
    print("=" * 50)
    
    # Step 2: Train the model automatically
    print("🧠 Step 2: Training the model...")
    
    try:
        # Change to parent directory where initial_model.py is located
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Run the initial model training
        print("🔄 Running initial_model.py...")
        result = subprocess.run([
            'python3', 
            os.path.join(parent_dir, 'initial_model.py')
        ], cwd=parent_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            print("📈 Training output:")
            print(result.stdout[-500:])  # Show last 500 chars of output
        else:
            print("❌ Model training failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False
    
    print("=" * 50)
    
    # Step 3: Copy trained model to deployment location
    print("📦 Step 3: Deploying trained model...")
    
    try:
        # Source: where initial_model.py saves the model
        source_model_dir = os.path.join(parent_dir, "forecasting_model")
        
        # Destination: where the monitoring system expects it
        dest_model_dir = "/opt/anomaly_detection/models/forecasting_model"
        
        if os.path.exists(source_model_dir):
            # Copy the entire model directory
            import shutil
            if os.path.exists(dest_model_dir):
                shutil.rmtree(dest_model_dir)
            shutil.copytree(source_model_dir, dest_model_dir)
            print(f"✅ Model deployed to {dest_model_dir}")
        else:
            print(f"⚠️ Source model directory not found: {source_model_dir}")
            return False
            
    except Exception as e:
        print(f"❌ Model deployment error: {e}")
        return False
    
    print("=" * 50)
    
    # Step 4: Verify everything is ready
    print("🔍 Step 4: Verification...")
    
    # Check if model files exist
    model_file = os.path.join(dest_model_dir, "model.h5")
    scalers_file = os.path.join(dest_model_dir, "scalers.pkl")
    variables_file = os.path.join(dest_model_dir, "variables.txt")
    
    files_ok = True
    for file_path, name in [(model_file, "Model"), (scalers_file, "Scalers"), (variables_file, "Variables")]:
        if os.path.exists(file_path):
            print(f"✅ {name} file exists")
        else:
            print(f"❌ {name} file missing: {file_path}")
            files_ok = False
    
    if files_ok:
        print("🎉 Complete Auto-Training SUCCESS!")
        print("")
        print("🚀 Ready to start monitoring:")
        print("   sudo systemctl start zabbix-monitoring")
        print("   sudo journalctl -u zabbix-monitoring -f")
        return True
    else:
        print("❌ Auto-training completed but some files are missing")
        return False


def quick_retrain():
    """Quick retrain with current Zabbix data"""
    print("⚡ Quick Retrain Mode...")
    return auto_train_model()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-Train Model with Current Zabbix Data')
    parser.add_argument('--config', '-c', default='config.json', help='Config file')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick retrain mode')
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_retrain()
    else:
        success = auto_train_model(args.config)
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Start monitoring: sudo systemctl start zabbix-monitoring")
        print("2. View logs: sudo journalctl -u zabbix-monitoring -f")
        print("3. Open dashboard: http://localhost:8050")
    else:
        print("\n❌ Auto-training failed - check the errors above")
        sys.exit(1)