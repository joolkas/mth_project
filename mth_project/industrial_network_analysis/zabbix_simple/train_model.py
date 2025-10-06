#!/usr/bin/env python3
"""
Simple Model Trainer for Zabbix Integration
Creates and trains LSTM forecasting model using collected data
"""

import json
import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import sys

# Add parent directory to import existing modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Import existing model functions
    from initial_model import create_online_multistep_model, split_data_for_multistep_model, train_model
    print("✅ Using existing model architecture")
    
except ImportError as e:
    print(f"⚠️ Could not import TensorFlow or existing models: {e}")
    print("Please install: pip install tensorflow")
    sys.exit(1)


class SimpleModelTrainer:
    """Simple model trainer using existing architecture"""
    
    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # Model parameters
        self.context_length = self.config['model']['context_length']
        self.prediction_horizon = self.config['model']['prediction_horizon']
        self.model_path = self.config['model']['model_path']
        
        # Training parameters
        self.epochs = 50
        self.batch_size = 32
        self.validation_split = 0.2
        
        # Create model directory
        os.makedirs(self.model_path, exist_ok=True)
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration"""
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_and_preprocess_data(self, data_file: str) -> pd.DataFrame:
        """Load and preprocess training data"""
        try:
            self.logger.info(f"Loading data from: {data_file}")
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            
            # Validate data
            if df.empty:
                raise ValueError("Data file is empty")
            
            if len(df) < self.context_length + self.prediction_horizon:
                raise ValueError(f"Need at least {self.context_length + self.prediction_horizon} records")
            
            # Select numeric columns only
            df_numeric = df.select_dtypes(include=[np.number])
            
            # Remove columns with insufficient variance
            df_clean = df_numeric.loc[:, df_numeric.std() > 1e-6]
            
            # Handle missing values
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Apply differencing (like existing system)
            df_differenced = df_clean.diff().dropna()
            
            self.logger.info(f"Data preprocessed: {df_differenced.shape}")
            return df_differenced
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def train_model(self, df: pd.DataFrame):
        """Train the forecasting model"""
        try:
            self.logger.info("Creating and training model...")
            
            # Create model using existing architecture
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
            
            # Prepare training data
            X_train, y_train, scalers, _ = split_data_for_multistep_model(
                df, self.context_length, self.prediction_horizon
            )
            
            self.logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
            
            # Train model
            model_save_path = os.path.join(self.model_path, "initial_model.h5")
            history, trained_model = train_model(
                model, X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                model_save_path=model_save_path
            )
            
            self.logger.info("Model training completed")
            
            # Save model and data
            self._save_model_files(trained_model, scalers, df, history)
            
            return trained_model, scalers
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise
    
    def _save_model_files(self, model, scalers, df, history):
        """Save all required model files"""
        try:
            self.logger.info("Saving model files...")
            
            # Save model
            model.save(os.path.join(self.model_path, "initial_model.h5"))
            
            # Save scalers
            with open(os.path.join(self.model_path, "scalers_train.pkl"), 'wb') as f:
                pickle.dump(scalers, f)
            
            # Save training data for online learning
            variables = df.columns.tolist()
            online_split = int(0.9 * len(df))
            df_online = df.iloc[online_split:].copy()
            
            df_online.to_csv(os.path.join(self.model_path, "df_online.csv"))
            df.to_csv(os.path.join(self.model_path, "df_removed_nans_forecasting.csv"))
            df.to_csv(os.path.join(self.model_path, "df_removed_nans_classification.csv"))
            
            # Save parameters
            np.save(os.path.join(self.model_path, "context_length.npy"), self.context_length)
            np.save(os.path.join(self.model_path, "model_mode.npy"), "multi_step")
            
            with open(os.path.join(self.model_path, "variables.txt"), 'w') as f:
                for var in variables:
                    f.write(f"{var}\n")
            
            # Save metadata
            metadata = {
                'model_mode': 'multi_step',
                'context_length': self.context_length,
                'prediction_horizon': self.prediction_horizon,
                'num_features': len(variables),
                'variables': variables,
                'trained_at': datetime.now().isoformat()
            }
            
            with open(os.path.join(self.model_path, "model_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model saved to: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM Model for Zabbix Integration')
    parser.add_argument('--data', '-d', required=True, help='Training data CSV file')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    print("🤖 Training Model for Zabbix Integration")
    print("=" * 50)
    
    try:
        trainer = SimpleModelTrainer()
        trainer.epochs = args.epochs
        trainer.batch_size = args.batch_size
        
        # Load and preprocess data
        df = trainer.load_and_preprocess_data(args.data)
        
        # Train model
        model, scalers = trainer.train_model(df)
        
        print("=" * 50)
        print("✅ Model training completed successfully!")
        print(f"Model saved to: {trainer.model_path}")
        print("Ready for real-time monitoring")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())