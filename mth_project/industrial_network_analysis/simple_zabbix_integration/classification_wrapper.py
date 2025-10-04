#!/usr/bin/env python3
"""
🔧 Classification Model Integration for Simplified Zabbix System

Uses your existing classification model without any changes.
This is a wrapper to integrate the classification system with the new Zabbix setup.

Author: Industrial Network Analysis System
Version: 1.0
"""

import sys
import os
import logging
from typing import Dict, Any, Optional

# Add parent directory to import existing modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    # Import your existing classification functions
    from classification_model import *  # Import all classification functions
    import pickle
    import numpy as np
    import pandas as pd
    from tensorflow import keras
except ImportError as e:
    print(f"❌ Cannot import classification modules: {e}")
    sys.exit(1)


class ZabbixClassificationModel:
    """Wrapper for your existing classification model"""
    
    def __init__(self, model_path: str = None):
        self.logger = self._setup_logging()
        
        # Default path to your classification model
        if model_path is None:
            model_path = os.path.join(parent_dir, "classification_model")
        
        self.model_path = model_path
        self.model = None
        self.preprocessing_objects = None
        
        # Load the model and preprocessing objects
        self._load_model()
    
    def _setup_logging(self):
        """Setup logging"""
        return logging.getLogger(__name__)
    
    def _load_model(self):
        """Load your existing trained classification model"""
        try:
            self.logger.info(f"🤖 Loading classification model from: {self.model_path}")
            
            # Try to find the classification model file
            model_candidates = [
                "improved_cnn_classifier.h5",
                "cnn_classifier.h5", 
                "final_model.h5",
                "best_model.h5"
            ]
            
            model_file = None
            for candidate in model_candidates:
                candidate_path = os.path.join(self.model_path, candidate)
                if os.path.exists(candidate_path):
                    model_file = candidate_path
                    break
            
            if model_file is None:
                raise FileNotFoundError(f"No classification model found in {self.model_path}")
            
            # Load the model
            self.model = keras.models.load_model(model_file)
            self.logger.info(f"✅ Classification model loaded: {os.path.basename(model_file)}")
            
            # Load preprocessing objects
            preprocessing_candidates = [
                "improved_preprocessing_objects.pkl",
                "preprocessing_objects.pkl",
                "encoders.pkl"
            ]
            
            preprocessing_file = None
            for candidate in preprocessing_candidates:
                candidate_path = os.path.join(self.model_path, candidate)
                if os.path.exists(candidate_path):
                    preprocessing_file = candidate_path
                    break
            
            if preprocessing_file is None:
                raise FileNotFoundError(f"No preprocessing objects found in {self.model_path}")
            
            with open(preprocessing_file, 'rb') as f:
                self.preprocessing_objects = pickle.load(f)
            
            self.logger.info(f"✅ Preprocessing objects loaded: {os.path.basename(preprocessing_file)}")
            
            # Validate required keys
            required_keys = ['label_to_index', 'label_to_name']
            for key in required_keys:
                if key not in self.preprocessing_objects:
                    raise KeyError(f"Missing required key '{key}' in preprocessing objects")
            
            # Create index_to_label if not present
            if 'index_to_label' not in self.preprocessing_objects:
                self.preprocessing_objects['index_to_label'] = {
                    v: k for k, v in self.preprocessing_objects['label_to_index'].items()
                }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load classification model: {e}")
            raise
    
    def predict(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Predict anomaly classification using your existing model
        
        Args:
            data: Input data for classification (should match your model's expected format)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if self.model is None:
                raise ValueError("Classification model not loaded")
            
            # Make prediction
            predictions = self.model.predict(data, verbose=0)
            
            # Get the predicted class
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            confidence = float(np.max(predictions))
            
            # Map to class labels
            predicted_label = self.preprocessing_objects['index_to_label'][predicted_class_idx]
            predicted_name = self.preprocessing_objects['label_to_name'].get(predicted_label, predicted_label)
            
            # Prepare result
            result = {
                'predicted_class': predicted_label,
                'predicted_name': predicted_name,
                'confidence': confidence,
                'class_probabilities': {
                    self.preprocessing_objects['index_to_label'][i]: float(prob)
                    for i, prob in enumerate(predictions[0])
                },
                'raw_predictions': predictions[0].tolist()
            }
            
            self.logger.debug(f"Classification result: {predicted_name} (confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Classification prediction failed: {e}")
            return {
                'predicted_class': 'unknown',
                'predicted_name': 'Unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {'model_loaded': False}
        
        return {
            'model_loaded': True,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_classes': len(self.preprocessing_objects['label_to_index']),
            'class_names': list(self.preprocessing_objects['label_to_name'].keys()),
            'model_path': self.model_path
        }
    
    def prepare_data_for_classification(self, forecasting_data: pd.DataFrame, 
                                      prediction_errors: np.ndarray = None) -> Optional[np.ndarray]:
        """
        Prepare data for classification using your existing preprocessing methods
        
        This is a simplified version - you may need to adapt based on your exact
        classification model input requirements.
        
        Args:
            forecasting_data: Time series data from forecasting
            prediction_errors: Prediction errors from forecasting model
            
        Returns:
            Prepared data for classification model
        """
        try:
            # This is a simplified example - adapt based on your classification model's
            # actual input requirements from your classification_model.py
            
            if forecasting_data is None or forecasting_data.empty:
                return None
            
            # Example preparation (adapt to your actual model requirements)
            # Your classification model might expect different input format
            
            # Option 1: Use prediction errors directly
            if prediction_errors is not None:
                # Reshape for CNN input (assuming 1D CNN)
                data = prediction_errors.reshape(1, -1, 1)
                return data
            
            # Option 2: Use recent time series data
            # Take the most recent data and reshape for your model
            recent_data = forecasting_data.tail(100).values  # Adjust window size as needed
            
            # Reshape for CNN (assuming your model expects this format)
            # Adjust dimensions based on your actual model requirements
            if len(recent_data.shape) == 2:
                data = recent_data.reshape(1, recent_data.shape[0], recent_data.shape[1])
            else:
                data = recent_data.reshape(1, -1, 1)
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Data preparation for classification failed: {e}")
            return None
    
    def classify_anomaly(self, forecasting_data: pd.DataFrame, 
                        prediction_errors: np.ndarray = None) -> Dict[str, Any]:
        """
        Complete anomaly classification pipeline
        
        Args:
            forecasting_data: Time series data from forecasting
            prediction_errors: Prediction errors from forecasting model
            
        Returns:
            Classification results
        """
        try:
            # Prepare data for classification
            prepared_data = self.prepare_data_for_classification(forecasting_data, prediction_errors)
            
            if prepared_data is None:
                return {
                    'predicted_class': 'unknown',
                    'predicted_name': 'Data Preparation Failed',
                    'confidence': 0.0,
                    'error': 'Could not prepare data for classification'
                }
            
            # Run classification
            result = self.predict(prepared_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Anomaly classification failed: {e}")
            return {
                'predicted_class': 'error',
                'predicted_name': 'Classification Error',
                'confidence': 0.0,
                'error': str(e)
            }


def test_classification_model(model_path: str = None):
    """Test the classification model"""
    print("🧪 Testing Classification Model Integration")
    print("=" * 50)
    
    try:
        # Initialize classification model
        classifier = ZabbixClassificationModel(model_path)
        
        # Get model info
        info = classifier.get_model_info()
        print(f"✅ Model loaded successfully:")
        print(f"   Model path: {info['model_path']}")
        print(f"   Input shape: {info['input_shape']}")
        print(f"   Output shape: {info['output_shape']}")
        print(f"   Number of classes: {info['num_classes']}")
        print(f"   Classes: {info['class_names']}")
        
        # Test with dummy data
        print("\n🔬 Testing with dummy data...")
        
        # Create dummy forecasting data (adapt dimensions as needed)
        dummy_data = pd.DataFrame(
            np.random.randn(100, 10),  # 100 time steps, 10 features
            columns=[f'var_{i}' for i in range(10)]
        )
        
        # Test classification
        result = classifier.classify_anomaly(dummy_data)
        
        print(f"📊 Test classification result:")
        print(f"   Predicted class: {result.get('predicted_class', 'N/A')}")
        print(f"   Predicted name: {result.get('predicted_name', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
        
        if 'class_probabilities' in result:
            print("   Class probabilities:")
            for class_name, prob in result['class_probabilities'].items():
                print(f"     {class_name}: {prob:.3f}")
        
        print("\n✅ Classification model test completed successfully!")
        
    except Exception as e:
        print(f"❌ Classification model test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Classification Model Integration')
    parser.add_argument('--model-path', '-m', help='Path to classification model directory')
    
    args = parser.parse_args()
    
    test_classification_model(args.model_path)


if __name__ == "__main__":
    main()