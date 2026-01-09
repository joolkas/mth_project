"""
Configuration Management Utility for MTH Project

This module provides a centralized way to manage paths and configuration
settings across the project, replacing hardcoded paths and making the code
more portable across different systems.
"""

import os
from pathlib import Path
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for the MTH project.
    
    Provides centralized management of paths, directories, and settings
    to improve code portability and maintainability.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize configuration.
        
        Args:
            config_file (str, optional): Path to JSON config file. 
                                        If None, uses default settings.
        """
        # Base paths - relative to this file's location
        self.PROJECT_ROOT = Path(__file__).parent.absolute()
        self.MTH_PROJECT_DIR = self.PROJECT_ROOT / "mth_project"
        self.INDUSTRIAL_ANALYSIS_DIR = self.MTH_PROJECT_DIR / "industrial_network_analysis"
        
        # Data directories
        self.DATA_DIR = self.INDUSTRIAL_ANALYSIS_DIR / "Data092025"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        
        # Model directories
        self.FORECASTING_MODEL_DIR = self.INDUSTRIAL_ANALYSIS_DIR / "forecasting_model"
        self.CLASSIFICATION_MODEL_DIR = self.INDUSTRIAL_ANALYSIS_DIR / "classification_model"
        
        # Log files
        self.LOG_DIR = self.PROJECT_ROOT
        self.WARNING_LOG = self.LOG_DIR / "warnings.log"
        self.DATASET_LOG = self.LOG_DIR / "dataset.log"
        
        # Default device name
        self.DEFAULT_DEVICE = "A1-SW-B-246"
        
        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file):
        """
        Load configuration from JSON file.
        
        Args:
            config_file (str): Path to JSON configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {config_file}: {e}")
            print(f"Error: Invalid JSON in config file {config_file}: {e}")
            return
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
            print(f"Error loading config file {config_file}: {e}")
            return
        
        # Update paths if specified in config
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, Path(value))
    
    def save_to_file(self, config_file):
        """
        Save current configuration to JSON file.
        
        Args:
            config_file (str): Path where to save configuration
        """
        config_data = {
            'DATA_DIR': str(self.DATA_DIR),
            'FORECASTING_MODEL_DIR': str(self.FORECASTING_MODEL_DIR),
            'CLASSIFICATION_MODEL_DIR': str(self.CLASSIFICATION_MODEL_DIR),
            'DEFAULT_DEVICE': self.DEFAULT_DEVICE,
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def ensure_directories_exist(self):
        """
        Create necessary directories if they don't exist.
        """
        dirs_to_create = [
            self.DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.FORECASTING_MODEL_DIR,
            self.CLASSIFICATION_MODEL_DIR,
            self.LOG_DIR,
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_device_data_path(self, device_name=None):
        """
        Get path to device CSV file.
        
        Args:
            device_name (str, optional): Name of device. Uses DEFAULT_DEVICE if None.
            
        Returns:
            Path: Path to device CSV file
        """
        if device_name is None:
            device_name = self.DEFAULT_DEVICE
        return self.DATA_DIR / f"{device_name}.csv"
    
    def get_processed_forecasting_path(self, device_name=None):
        """
        Get path to processed forecasting data.
        
        Args:
            device_name (str, optional): Name of device. Uses DEFAULT_DEVICE if None.
            
        Returns:
            Path: Path to processed forecasting CSV
        """
        if device_name is None:
            device_name = self.DEFAULT_DEVICE
        return self.PROCESSED_DATA_DIR / f"{device_name}_forecasting.csv"
    
    def get_processed_statuses_path(self, device_name=None):
        """
        Get path to processed statuses data.
        
        Args:
            device_name (str, optional): Name of device. Uses DEFAULT_DEVICE if None.
            
        Returns:
            Path: Path to processed statuses CSV
        """
        if device_name is None:
            device_name = self.DEFAULT_DEVICE
        return self.PROCESSED_DATA_DIR / f"{device_name}_statuses.csv"
    
    def get_classification_scaler_path(self):
        """
        Get path to classification scaler pickle file.
        
        Returns:
            Path: Path to scaler.pkl
        """
        return self.CLASSIFICATION_MODEL_DIR / "scaler.pkl"
    
    def get_classification_model_path(self, model_name="final_model.h5"):
        """
        Get path to classification model file.
        
        Args:
            model_name (str): Name of model file
            
        Returns:
            Path: Path to model file
        """
        return self.CLASSIFICATION_MODEL_DIR / model_name
    
    def get_classification_encoders_path(self):
        """
        Get path to classification encoders pickle file.
        
        Returns:
            Path: Path to encoders.pkl
        """
        return self.CLASSIFICATION_MODEL_DIR / "encoders.pkl"
    
    def __str__(self):
        """String representation of configuration."""
        return f"""MTH Project Configuration:
  Project Root: {self.PROJECT_ROOT}
  Data Directory: {self.DATA_DIR}
  Forecasting Model Dir: {self.FORECASTING_MODEL_DIR}
  Classification Model Dir: {self.CLASSIFICATION_MODEL_DIR}
  Default Device: {self.DEFAULT_DEVICE}
"""


# Global configuration instance
config = Config()


if __name__ == "__main__":
    # Demo usage
    print("MTH Project Configuration Manager")
    print("=" * 50)
    print(config)
    
    print("\nExample paths:")
    print(f"Device data: {config.get_device_data_path()}")
    print(f"Processed forecasting: {config.get_processed_forecasting_path()}")
    print(f"Classification model: {config.get_classification_model_path()}")
    
    # Optionally create directories
    print("\nCreating necessary directories...")
    config.ensure_directories_exist()
    print("Done!")
