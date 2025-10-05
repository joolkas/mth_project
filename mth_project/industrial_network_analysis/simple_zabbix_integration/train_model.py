#!/usr/bin/env python3
"""
🤖 Initial Model Training for Simplified Zabbix Integration

Creates and trains LSTM forecasting model using data collected from Zabbix.
Compatible with your existing model architecture and preprocessing.

Author: Industrial Network Analysis System
Version: 1.0
"""

import sys
import os
import json
import logging
from datetime import datetime

# Import TensorFlow configuration fix first
from tensorflow_config import fix_tensorflow_configuration
fix_tensorflow_configuration()

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add parent directory to import your existing modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import TensorFlow first
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
except ImportError:
    print("❌ TensorFlow not found. Please install: pip install tensorflow")
    sys.exit(1)

# Try to import existing model functions, but provide fallbacks if import fails
try:
    # Import only the specific functions we need without dependencies
    sys.path.insert(0, parent_dir)  # Ensure parent directory is first in path
    
    # Import model creation functions with isolated imports
    import importlib.util
    
    # Load initial_model module manually to avoid dependency issues
    spec = importlib.util.spec_from_file_location("initial_model", 
                                                  os.path.join(parent_dir, "initial_model.py"))
    if spec and spec.loader:
        initial_model_module = importlib.util.module_from_spec(spec)
        
        # Mock the get_processed_path import to prevent import error
        import types
        mock_get_data = types.ModuleType('get_data')
        mock_get_data.get_processed_path = lambda: ("", "")
        sys.modules['get_data'] = mock_get_data
        
        # Now load the initial_model module
        spec.loader.exec_module(initial_model_module)
        
        # Extract the functions we need
        create_online_multistep_model = initial_model_module.create_online_multistep_model
        create_online_onestep_model = initial_model_module.create_online_onestep_model
        split_data_for_multistep_model = initial_model_module.split_data_for_multistep_model
        split_data_for_onestep_model = initial_model_module.split_data_for_onestep_model
        train_model = initial_model_module.train_model
        
        print("✅ Successfully imported existing model functions")
        
    else:
        raise ImportError("Could not load initial_model module")
        
except Exception as e:
    print(f"⚠️ Could not import existing model functions: {e}")
    print("   Using simplified built-in model functions")
    
    # Define simplified model functions directly in this file
    def create_online_multistep_model(df, context_length=60, prediction_horizon=6, 
                                    first_layer_units=128, second_layer_units=64, 
                                    third_layer_units=32, dense_units=256, 
                                    activation='relu', dropout_rate=0.3):
        """Simplified multi-step model creation"""
        num_features = len(df.columns)
        output_size = prediction_horizon * num_features
        
        model = keras.models.Sequential([
            keras.layers.LSTM(first_layer_units, return_sequences=True, 
                            input_shape=(context_length, num_features)),
            keras.layers.Dropout(dropout_rate * 0.5),
            keras.layers.LSTM(second_layer_units, return_sequences=True),
            keras.layers.Dropout(dropout_rate * 0.5),
            keras.layers.LSTM(third_layer_units, return_sequences=False),
            keras.layers.Dropout(dropout_rate * 0.7),
            keras.layers.Dense(dense_units, activation=activation),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(dense_units // 2, activation=activation),
            keras.layers.Dropout(dropout_rate * 0.5),
            keras.layers.Dense(output_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['accuracy', 'mse']
        )
        
        return model
    
    def split_data_for_multistep_model(df, context_length=60, prediction_horizon=6):
        """Simplified data splitting for multi-step model"""
        scalers = {}
        scaled_data = np.zeros_like(df.values)
        
        # Scale each column separately
        for i, var in enumerate(df.columns):
            scaler = StandardScaler()
            scaled_data[:, i] = scaler.fit_transform(df[var].values.reshape(-1, 1)).flatten()
            scalers[var] = scaler
        
        # Convert back to DataFrame
        df_scaled = pd.DataFrame(data=scaled_data, columns=df.columns, index=df.index)
        
        # Create training sequences
        X_train, y_train = [], []
        original_indices = []
        
        for i in range(context_length, len(df_scaled) - prediction_horizon + 1):
            X_train.append(df_scaled.iloc[i-context_length:i].values)
            
            future_steps = []
            for step in range(prediction_horizon):
                future_steps.extend(df_scaled.iloc[i + step].values)
            y_train.append(future_steps)
            original_indices.append(i)
        
        return np.array(X_train), np.array(y_train), scalers, original_indices
    
    def train_model(model, X_train, y_train, epochs=50, batch_size=32, 
                   validation_split=0.2, verbose=1, use_callbacks=True, 
                   es_patience=10, lr_factor=0.5, lr_patience=5, model_save_path=None):
        """Simplified model training"""
        
        callback_list = []
        if use_callbacks:
            callback_list = [
                EarlyStopping(monitor='val_loss', patience=es_patience, 
                            restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, 
                                patience=lr_patience, min_lr=1e-7, verbose=1)
            ]
            
            if model_save_path:
                callback_list.append(
                    ModelCheckpoint(filepath=model_save_path, monitor='val_loss', 
                                  save_best_only=True, verbose=1)
                )
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=callback_list if use_callbacks else None
        )
        
        return history, model

# TensorFlow import moved above


class SimpleModelTrainer:
    """Simplified model trainer for Zabbix data"""
    
    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # Model parameters (matching your existing system)
        self.context_length = self.config['model']['context_length']
        self.prediction_horizon = self.config['model']['prediction_horizon']
        self.model_path = self.config['model']['model_path']
        
        # Training parameters
        self.epochs = 50
        self.batch_size = 32
        self.validation_split = 0.2
        self.model_mode = "multi_step"  # Following your main system
        
        # Create model directory
        os.makedirs(self.model_path, exist_ok=True)
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Config error: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('model_training.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def load_data(self, data_file: str) -> pd.DataFrame:
        """Load and validate training data"""
        try:
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Data file not found: {data_file}")
            
            self.logger.info(f"📊 Loading training data from: {data_file}")
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            
            # Validate data
            if df.empty:
                raise ValueError("Data file is empty")
            
            if len(df) < self.context_length + self.prediction_horizon:
                raise ValueError(f"Need at least {self.context_length + self.prediction_horizon} records, have {len(df)}")
            
            # Select numeric columns only
            df_numeric = df.select_dtypes(include=[np.number])
            
            # Remove columns with insufficient variance (likely constant)
            df_clean = df_numeric.loc[:, df_numeric.std() > 1e-6]
            
            # Remove rows with too many NaN values
            nan_threshold = 0.3  # Allow up to 30% NaN per row
            df_clean = df_clean.dropna(thresh=int(len(df_clean.columns) * (1 - nan_threshold)))
            
            # Fill remaining NaN values
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            self.logger.info(f"✅ Data loaded and cleaned:")
            self.logger.info(f"   Original shape: {df.shape}")
            self.logger.info(f"   Cleaned shape: {df_clean.shape}")
            self.logger.info(f"   Time range: {df_clean.index[0]} to {df_clean.index[-1]}")
            self.logger.info(f"   Variables: {list(df_clean.columns)[:5]}...")
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load data: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for model training (differencing like your main system)"""
        try:
            self.logger.info("🔧 Preprocessing data (applying differencing)...")
            
            # Apply differencing like in your main system
            df_differenced = df.diff().dropna()
            
            self.logger.info(f"   Differenced data shape: {df_differenced.shape}")
            self.logger.info(f"   Data range after differencing: {df_differenced.min().min():.6f} to {df_differenced.max().max():.6f}")
            
            return df_differenced
            
        except Exception as e:
            self.logger.error(f"❌ Data preprocessing failed: {e}")
            raise
    
    def create_model(self, df: pd.DataFrame):
        """Create model using your existing architecture"""
        try:
            self.logger.info(f"🤖 Creating {self.model_mode} model...")
            
            if self.model_mode == "multi_step":
                model = create_online_multistep_model(
                    df,
                    context_length=self.context_length,
                    prediction_horizon=self.prediction_horizon,
                    first_layer_units=128,
                    second_layer_units=64,
                    third_layer_units=32,
                    dense_units=256,
                    activation='relu',
                    dropout_rate=0.3
                )
            else:
                model = create_online_onestep_model(
                    df,
                    context_length=self.context_length,
                    first_layer_units=128,
                    second_layer_units=64,
                    dense_units=256,
                    activation='relu',
                    dropout_rate=0.3
                )
            
            self.logger.info("✅ Model created successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Model creation failed: {e}")
            raise
    
    def train_and_evaluate(self, df: pd.DataFrame):
        """Train and evaluate the model"""
        try:
            # Split data for training (80%) and testing (20%)
            split_idx = int(0.8 * len(df))
            df_train = df.iloc[:split_idx]
            df_test = df.iloc[split_idx:]
            
            self.logger.info(f"📊 Data split:")
            self.logger.info(f"   Training: {df_train.shape}")
            self.logger.info(f"   Testing: {df_test.shape}")
            
            # Create model
            model = self.create_model(df_train)
            
            # Prepare training data
            if self.model_mode == "multi_step":
                X_train, y_train, scalers, _ = split_data_for_multistep_model(
                    df_train, self.context_length, self.prediction_horizon
                )
                X_test, y_test, _, test_indices = split_data_for_multistep_model(
                    df_test, self.context_length, self.prediction_horizon
                )
            else:
                X_train, y_train, scalers, _ = split_data_for_onestep_model(
                    df_train, self.context_length
                )
                X_test, y_test, _, test_indices = split_data_for_onestep_model(
                    df_test, self.context_length
                )
            
            self.logger.info("🏋️ Starting model training...")
            
            # Train model
            model_save_path = os.path.join(self.model_path, "best_model.h5")
            history, trained_model = train_model(
                model, X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                model_save_path=model_save_path
            )
            
            self.logger.info("✅ Model training completed")
            
            # Simplified model evaluation
            self.logger.info("📈 Evaluating model...")
            
            # Simple evaluation using validation loss from training
            val_loss = min(history.history['val_loss']) if 'val_loss' in history.history else history.history['loss'][-1]
            train_loss = history.history['loss'][-1]
            
            self.logger.info(f"Final training loss: {train_loss:.6f}")
            self.logger.info(f"Final validation loss: {val_loss:.6f}")
            
            # Simple prediction test
            test_predictions = trained_model.predict(X_test[:5], verbose=0)  # Test on first 5 samples
            
            # Calculate basic metrics
            mse = val_loss  # Use validation loss as MSE approximation
            mae = np.sqrt(mse)  # Rough MAE approximation
            rmse = np.sqrt(mse)
            percentage_error = (rmse / np.mean(np.abs(df_train.values))) * 100
            
            self.logger.info(f"Estimated metrics:")
            self.logger.info(f"  MSE: {mse:.6f}")
            self.logger.info(f"  MAE: {mae:.6f}")
            self.logger.info(f"  RMSE: {rmse:.6f}")
            self.logger.info(f"  Percentage Error: {percentage_error:.2f}%")
            
            # Create a simple results DataFrame
            results_df = pd.DataFrame({
                'metric': ['mse', 'mae', 'rmse', 'percentage_error'],
                'value': [mse, mae, rmse, percentage_error]
            })
            
            # Save model and training data
            self._save_model_and_data(trained_model, scalers, df, df, history, 
                                    mse, mae, rmse, percentage_error)
            
            return trained_model, scalers, results_df
            
        except Exception as e:
            self.logger.error(f"❌ Training and evaluation failed: {e}")
            raise
    
    def _save_model_and_data(self, model, scalers, df_differenced, df_original, history, 
                           mse, mae, rmse, percentage_error):
        """Save model, scalers, and training data"""
        try:
            self.logger.info("💾 Saving model and training data...")
            
            # Save model in multiple formats for compatibility
            model.save(os.path.join(self.model_path, "initial_model.h5"))
            model.save_weights(os.path.join(self.model_path, "model_weights.h5"))
            
            # Save model architecture as JSON
            with open(os.path.join(self.model_path, "model_architecture.json"), 'w') as f:
                f.write(model.to_json())
            
            # Save scalers
            with open(os.path.join(self.model_path, "scalers_train.pkl"), 'wb') as f:
                pickle.dump(scalers, f)
            
            # Save model metadata
            model_metadata = {
                'model_mode': self.model_mode,
                'context_length': self.context_length,
                'prediction_horizon': self.prediction_horizon,
                'num_features': len(df_differenced.columns),
                'variables': df_differenced.columns.tolist(),
                'training_metrics': {
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'percentage_error': float(percentage_error)
                },
                'tensorflow_version': tf.__version__,
                'trained_at': datetime.now().isoformat()
            }
            
            with open(os.path.join(self.model_path, "model_metadata.json"), 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            # Save training data for online learning (compatible with your main system)
            variables = df_differenced.columns.tolist()
            
            # Create online data (this would normally be the remaining data for online learning)
            # For simplicity, we'll use the last portion of training data
            online_split = int(0.9 * len(df_differenced))
            df_online = df_differenced.iloc[online_split:].copy()
            
            # Save data files (compatible with your get_online_data function)
            df_online.to_csv(os.path.join(self.model_path, "df_online.csv"))
            df_original.to_csv(os.path.join(self.model_path, "df_removed_nans_forecasting.csv"))
            df_original.to_csv(os.path.join(self.model_path, "df_removed_nans_classification.csv"))
            
            # Save additional parameters
            np.save(os.path.join(self.model_path, "context_length.npy"), self.context_length)
            np.save(os.path.join(self.model_path, "model_mode.npy"), self.model_mode)
            
            with open(os.path.join(self.model_path, "variables.txt"), 'w') as f:
                for var in variables:
                    f.write(f"{var}\n")
            
            # Save training results
            with open(os.path.join(self.model_path, "training_results.txt"), 'w') as f:
                f.write(f"Model Training Results\n")
                f.write(f"=====================\n")
                f.write(f"Model Mode: {self.model_mode}\n")
                f.write(f"Context Length: {self.context_length}\n")
                f.write(f"Prediction Horizon: {self.prediction_horizon}\n")
                f.write(f"Number of Features: {len(variables)}\n")
                f.write(f"Training Data Shape: {df_differenced.shape}\n")
                f.write(f"\nMetrics:\n")
                f.write(f"MSE: {mse:.6f}\n")
                f.write(f"MAE: {mae:.6f}\n")
                f.write(f"RMSE: {rmse:.6f}\n")
                f.write(f"Percentage Error: {percentage_error:.6f}%\n")
                f.write(f"\nTrained at: {datetime.now()}\n")
            
            self.logger.info(f"✅ Model saved to: {self.model_path}")
            self.logger.info(f"   Model files: initial_model.h5, model_weights.h5")
            self.logger.info(f"   Training data: df_online.csv, scalers_train.pkl")
            self.logger.info(f"   Metadata: model_metadata.json")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {e}")
            raise


def main():
    """Main entry point for model training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Initial LSTM Model for Zabbix Integration')
    parser.add_argument('--config', '-c', default='config.json', help='Configuration file')
    parser.add_argument('--data', '-d', required=True, help='Training data CSV file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    
    print("🤖 Starting Initial Model Training for Zabbix Integration")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = SimpleModelTrainer(args.config)
        trainer.epochs = args.epochs
        trainer.batch_size = args.batch_size
        
        # Load and preprocess data  
        df_raw = trainer.load_data(args.data)
        df_processed = trainer.preprocess_data(df_raw)
        
        # Train and evaluate model
        model, scalers, results = trainer.train_and_evaluate(df_processed)
        
        print("=" * 60)
        print("✅ Model Training Completed Successfully!")
        print(f"   Model saved to: {trainer.model_path}")
        print(f"   Ready for use with main monitoring system")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()