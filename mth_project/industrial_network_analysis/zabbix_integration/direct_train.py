#!/usr/bin/env python3
"""
🤖 Complete Self-Contained Auto-Training

Trains the model directly without depending on old file paths.
Uses the collected Zabbix data to train a new model from scratch.
"""

import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_training import collect_training_data


def create_sequences(data, context_length, prediction_horizon):
    """Create sequences for training (same logic as initial_model.py)"""
    X, y = [], []
    
    for i in range(len(data) - context_length - prediction_horizon + 1):
        # Input sequence
        X.append(data[i:(i + context_length)])
        # Output sequence (multi-step prediction)
        y.append(data[(i + context_length):(i + context_length + prediction_horizon)])
    
    return np.array(X), np.array(y)


def build_model(num_features, context_length=60, prediction_horizon=6):
    """Build LSTM model (simplified version of initial_model.py)"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(context_length, num_features)),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(num_features * prediction_horizon),  # Multi-step output
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_model_direct(config_path="config.json"):
    """Train model directly from Zabbix data"""
    
    print("🤖 Self-Contained Auto-Training Pipeline...")
    print("=" * 60)
    
    # Step 1: Collect training data
    print("📊 Step 1: Collecting training data from Zabbix...")
    df_forecasting, df_classification = collect_training_data(config_path, days=7)
    
    if df_forecasting is None or df_forecasting.empty:
        print("❌ Training data collection failed!")
        return False
    
    print(f"✅ Data collected: {df_forecasting.shape}")
    print(f"📋 Features: {list(df_forecasting.columns)[:5]}...")
    print("=" * 60)
    
    # Step 2: Preprocessing
    print("🔄 Step 2: Preprocessing data...")
    
    # Remove any non-numeric columns and handle NaN values
    df_numeric = df_forecasting.select_dtypes(include=[np.number])
    df_clean = df_numeric.fillna(method='ffill').fillna(0)
    
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_clean)
    
    num_features = data_scaled.shape[1]
    context_length = 60
    prediction_horizon = 6
    
    print(f"✅ Preprocessed data: {data_scaled.shape}")
    print(f"📊 Features: {num_features}, Context: {context_length}, Horizon: {prediction_horizon}")
    print("=" * 60)
    
    # Step 3: Create sequences
    print("🔄 Step 3: Creating training sequences...")
    
    X, y = create_sequences(data_scaled, context_length, prediction_horizon)
    
    if len(X) == 0:
        print("❌ Not enough data to create sequences!")
        print(f"Need at least {context_length + prediction_horizon} samples, have {len(data_scaled)}")
        return False
    
    # Reshape y for multi-step prediction
    y = y.reshape(y.shape[0], -1)  # Flatten the multi-step output
    
    print(f"✅ Created sequences: X={X.shape}, y={y.shape}")
    print("=" * 60)
    
    # Step 4: Train model
    print("🧠 Step 4: Training the model...")
    
    try:
        model = build_model(num_features, context_length, prediction_horizon)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"📊 Training set: {X_train.shape}, Validation set: {X_val.shape}")
        
        # Train with early stopping
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # Reduced for faster training
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Model training completed!")
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 60)
    
    # Step 5: Save model and components
    print("💾 Step 5: Saving model...")
    
    try:
        # Create model directory
        model_dir = "/opt/anomaly_detection/models/forecasting_model"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_file = os.path.join(model_dir, "model.h5")
        model.save(model_file)
        print(f"✅ Model saved: {model_file}")
        
        # Save scaler
        scalers_dict = {}
        for i in range(num_features):
            scalers_dict[i] = scaler  # Use same scaler for all features
        
        scalers_file = os.path.join(model_dir, "scalers.pkl")
        with open(scalers_file, 'wb') as f:
            pickle.dump(scalers_dict, f)
        print(f"✅ Scalers saved: {scalers_file}")
        
        # Save variables
        variables_file = os.path.join(model_dir, "variables.txt")
        with open(variables_file, 'w') as f:
            for col in df_clean.columns:
                f.write(f"{col}\n")
        print(f"✅ Variables saved: {variables_file}")
        
        # Save model info
        info_file = os.path.join(model_dir, "model_info.json")
        model_info = {
            "num_features": num_features,
            "context_length": context_length, 
            "prediction_horizon": prediction_horizon,
            "training_samples": len(X),
            "input_shape": list(model.input_shape),
            "output_shape": list(model.output_shape)
        }
        
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"✅ Model info saved: {info_file}")
        
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
        return False
    
    print("=" * 60)
    
    # Step 6: Verification
    print("🔍 Step 6: Verification...")
    
    required_files = [
        (model_file, "Model"),
        (scalers_file, "Scalers"), 
        (variables_file, "Variables"),
        (info_file, "Model Info")
    ]
    
    all_good = True
    for file_path, name in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {name}: {file_path} ({size} bytes)")
        else:
            print(f"❌ {name}: Missing!")
            all_good = False
    
    if all_good:
        print("🎉 COMPLETE SUCCESS!")
        print("")  
        print("🚀 Ready to start monitoring:")
        print(f"   Model expects: {num_features} features")
        print(f"   Input shape: {model.input_shape}")
        print("   sudo systemctl start zabbix-monitoring")
        print("   sudo journalctl -u zabbix-monitoring -f")
        return True
    else:
        print("❌ Some components are missing!")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Self-Contained Auto-Training')
    parser.add_argument('--config', '-c', default='config.json', help='Config file')
    
    args = parser.parse_args()
    
    success = train_model_direct(args.config)
    
    if not success:
        print("\n❌ Auto-training failed!")
        sys.exit(1)
    else:
        print("\n🎯 Training complete - system ready to monitor!")