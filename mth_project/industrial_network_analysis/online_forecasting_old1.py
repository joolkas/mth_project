from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import time
import sys
import os
import pickle
from tensorflow.keras.models import load_model

from dash_plotter import DashRealTimePlotter

from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

# Try to import the improved classification model function
try:
    from classification_model_improved import create_multihot_encoding
except ImportError:
    print("Warning: classification_model_improved module not found. Classification features may not work.")
    create_multihot_encoding = None

# Global model storage to avoid repeated loading
_classification_model = None
_preprocessing_data = None

class OnlineForecastingSystem:
    """
    Enhanced online forecasting system with proper model management,
    error handling, and temporal consistency.
    """
    
    def __init__(self, models_dir=None):
        """
        Initialize the forecasting system with model paths.
        
        Args:
            models_dir: Directory containing trained models. If None, uses default path.
        """
        if models_dir is None:
            # Use relative path or allow configuration
            self.models_dir = os.path.join(
                os.path.dirname(__file__), 
                "..", "..", "experimenting", "03_Forecasting_and_classification_approach", "models"
            )
        else:
            self.models_dir = models_dir
            
        self.classification_model = None
        self.preprocessing_data = None
        self.is_initialized = False
        
    def initialize_classification_model(self):
        """
        Load classification model and preprocessing data once.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the model
            model_path = os.path.join(self.models_dir, "improved_cnn_classifier.h5")
            if not os.path.exists(model_path):
                # Fallback to original model
                model_path = os.path.join(self.models_dir, "cnn_classifier.h5")
                
            if not os.path.exists(model_path):
                print(f"Warning: No classification model found in {self.models_dir}")
                return False
                
            self.classification_model = load_model(model_path)
            
            # Load preprocessing objects
            preprocessing_path = os.path.join(self.models_dir, "improved_preprocessing_objects.pkl")
            if not os.path.exists(preprocessing_path):
                # Fallback to original preprocessing
                preprocessing_path = os.path.join(self.models_dir, "preprocessing_objects.pkl")
                
            if not os.path.exists(preprocessing_path):
                print(f"Warning: No preprocessing objects found in {self.models_dir}")
                return False
                
            with open(preprocessing_path, 'rb') as f:
                self.preprocessing_data = pickle.load(f)
            
            self.is_initialized = True
            print("Classification model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading classification model: {e}")
            return False

    def predict_recursive_steps(self, initial_model, initial_context, variables, prediction_horizon=6, context_length=60):
        """
        Predict multiple steps ahead using recursive feedback.
        
        Args:
            initial_model: Trained forecasting model
            initial_context: Initial context window (shape: context_length, num_features)
            variables: List of variable names being predicted
            prediction_horizon: Number of steps to predict ahead (default 6)
            context_length: Length of context window (default 60)
            
        Returns:
            predictions: List of predictions for each step
            
        Raises:
            ValueError: If input shapes don't match expected dimensions
        """
        # Input validation
        if initial_context is None or len(initial_context) == 0:
            raise ValueError("Initial context cannot be empty")
            
        if len(initial_context) != context_length:
            raise ValueError(f"Context length mismatch: expected {context_length}, got {len(initial_context)}")
            
        if len(variables) != initial_context.shape[1]:
            raise ValueError(f"Variables count mismatch: expected {len(variables)}, got {initial_context.shape[1]}")
        
        context = initial_context.copy()
        predictions = []
        
        try:
            for step in range(prediction_horizon):
                # Reshape for model input (batch_size=1, timesteps, features)
                context_reshaped = context.reshape(1, context_length, len(variables))
                step_prediction = initial_model.predict(context_reshaped, verbose=0)
                predictions.append(step_prediction.flatten())

                # Recursive feedback mechanism
                # Each prediction becomes input for the next prediction
                new_row = context[-1].copy()  # Start with last row of context
                
                # Update values with predictions (true feedback loop)
                for i in range(len(variables)):
                    new_row[i] = step_prediction[0, i]

                # Slide context window - remove oldest, add new prediction
                context = np.vstack((context[1:], new_row))

        except Exception as e:
            print(f"Error during recursive prediction at step {step}: {e}")
            raise
            
        return predictions

    def inverse_difference(self, predictions_arrays, last_actual_values):
        """
        Convert differenced predictions back to actual values.
        
        Args:
            predictions_arrays: List of prediction arrays (differenced values)
            last_actual_values: The last known actual values as reference point
            
        Returns:
            actual_predictions: List of actual value predictions
            
        Raises:
            ValueError: If input arrays have mismatched dimensions
        """
        if not predictions_arrays:
            return []
            
        if len(last_actual_values) != len(predictions_arrays[0]):
            raise ValueError("Dimension mismatch between last_actual_values and predictions")
        
        actual_predictions = []
        current_values = last_actual_values.copy()
        
        for pred_diff in predictions_arrays:
            try:
                # Add difference to get actual value
                current_values = current_values + np.array(pred_diff)
                actual_predictions.append(current_values.copy())
            except Exception as e:
                print(f"Error in inverse differencing: {e}")
                # Use previous values as fallback
                actual_predictions.append(current_values.copy())
        
        return actual_predictions

    def _perform_classification(self, step_predictions_actual, current_time_idx, 
                               df_removed_nans_classification, forecasting_variables):
        """
        Perform network status classification with proper temporal alignment.
        
        Args:
            step_predictions_actual: Actual predictions (non-differenced)
            current_time_idx: Current time index in the main loop
            df_removed_nans_classification: Classification data
            forecasting_variables: List of forecasting variable names
            
        Returns:
            str: Classification result or None if failed
        """
        try:
            # Get the required window size from preprocessing data
            window_size = self.preprocessing_data.get('window_size', len(step_predictions_actual))
            
            # Don't adjust window_size dynamically - this breaks the model
            if len(step_predictions_actual) != window_size:
                print(f"Warning: Prediction length {len(step_predictions_actual)} doesn't match expected window size {window_size}")
                # Pad or truncate to match expected size
                if len(step_predictions_actual) < window_size:
                    # Pad with last values
                    last_pred = step_predictions_actual[-1]
                    while len(step_predictions_actual) < window_size:
                        step_predictions_actual.append(last_pred)
                else:
                    # Truncate to required size
                    step_predictions_actual = step_predictions_actual[:window_size]

            # Create DataFrame with predictions
            temp_forecasting_df = pd.DataFrame(
                step_predictions_actual, 
                columns=forecasting_variables
            )

            # Get TEMPORALLY ALIGNED status data
            # Use status data that corresponds to the SAME time period as predictions
            status_start_idx = max(0, current_time_idx - window_size)
            status_end_idx = current_time_idx
            
            if status_end_idx > len(df_removed_nans_classification):
                # Use the last available window
                status_end_idx = len(df_removed_nans_classification)
                status_start_idx = max(0, status_end_idx - window_size)
            
            status_window = df_removed_nans_classification.iloc[status_start_idx:status_end_idx].copy()

            # Ensure same number of rows
            if len(status_window) != len(temp_forecasting_df):
                if len(status_window) < len(temp_forecasting_df):
                    # Pad status window with last row
                    last_row = status_window.iloc[-1:] if len(status_window) > 0 else None
                    while len(status_window) < len(temp_forecasting_df) and last_row is not None:
                        status_window = pd.concat([status_window, last_row], ignore_index=True)
                else:
                    # Truncate status window
                    status_window = status_window.iloc[-len(temp_forecasting_df):].copy()

            # Merge predictions with status data
            # Remove timestamp column from status if it exists
            status_cols_to_use = [col for col in status_window.columns if col != 'timestamp']
            combined_df = pd.concat([
                temp_forecasting_df.reset_index(drop=True),
                status_window[status_cols_to_use].reset_index(drop=True)
            ], axis=1)

            # Apply multi-hot encoding
            df_encoded, _ = create_multihot_encoding(combined_df)

            # Extract feature columns and reshape
            feature_columns = self.preprocessing_data['feature_columns']
            classification_input = df_encoded[feature_columns].values
            classification_input = classification_input.reshape(1, window_size, len(feature_columns))

            # Make prediction
            prediction_probs = self.classification_model.predict(classification_input, verbose=0)
            prediction_idx = prediction_probs.argmax(axis=1)[0]
            prediction_confidence = prediction_probs[0][prediction_idx]

            # Decode prediction
            class_names = self.preprocessing_data['class_names']
            prediction_name = class_names[prediction_idx]

            return f"{prediction_name} (confidence: {prediction_confidence:.2%})"
            
        except Exception as e:
            print(f"Classification error: {e}")
            return None
        
        def rolling_buffer_prediction_with_dash(self, initial_model, df_online, scalers_train,
                                            context_length, df_removed_nans_forecasting, df_removed_nans_classification, 
                                            dash_plotter, prediction_horizon=6, real_time_delay=0):

            # Input validation
            if len(df_online) < context_length + prediction_horizon:
                raise ValueError(f"Insufficient data: need at least {context_length + prediction_horizon} rows")
                
            # Initialize classification model if not already done
            if not self.is_initialized:
                classification_success = self.initialize_classification_model()
                if not classification_success:
                    print("Warning: Classification features disabled due to model loading failure")
            
            # Prepare data
            df_online_scaled = df_online.copy()
            forecasting_variables = df_online.columns.tolist()
            
            # Scale the online data
            for var in df_online.columns:
                if var in scalers_train:
                    scaler = scalers_train[var]
                    df_online_scaled[var] = scaler.transform(df_online[[var]])
                else:
                    print(f"Warning: No scaler found for variable {var}")
            
            # Initialize result containers
            final_predictions = []
            final_actuals = []
            final_timestamps = []
            predictions_actuals = []
            actuals_actuals = []
            
            total_steps = len(df_online_scaled) - context_length - prediction_horizon + 1

            if dash_plotter is not None:
                dash_plotter.set_total_steps(total_steps)

            # Initialize context window
            current_context = df_online_scaled[:context_length].copy().values

            print(f"Starting rolling prediction for {total_steps} steps...")
            
            # Main prediction loop
            for t in range(context_length, len(df_online_scaled) - prediction_horizon + 1):
                current_step = t - context_length
                
                try:
                    # 1. Make predictions for next 'prediction_horizon' steps
                    step_predictions = self.predict_recursive_steps(
                        initial_model, current_context, 
                        variables=forecasting_variables, 
                        prediction_horizon=prediction_horizon,
                        context_length=context_length
                    )
                    
                    # 2. Convert all predictions to original scale
                    step_predictions_original = []
                    for pred in step_predictions:
                        pred_original = []
                        for i, var in enumerate(forecasting_variables):
                            # Handle status variables specially
                            if 'status' in var.lower():
                                pred[i] = np.round(pred[i])
                            
                            if var in scalers_train:
                                scaler = scalers_train[var]
                                try:
                                    original_val = scaler.inverse_transform([[pred[i]]])[0, 0]
                                    pred_original.append(original_val)
                                except Exception as e:
                                    print(f"Warning: Error inverse transforming {var}: {e}")
                                    pred_original.append(pred[i])  # Use scaled value as fallback
                            else:
                                pred_original.append(pred[i])
                        step_predictions_original.append(pred_original)
                    
                    # 3. Get actual values for all predicted steps
                    actual_values = []
                    for step in range(prediction_horizon):
                        if t + step < len(df_online_scaled):
                            actual_values.append(df_online_scaled.iloc[t + step].values)

                    # Convert actuals to original scale
                    actuals_original = []
                    for actual in actual_values:
                        actual_original = []
                        for i, var in enumerate(forecasting_variables):
                            if var in scalers_train:
                                scaler = scalers_train[var]
                                try:
                                    original_val = scaler.inverse_transform([[actual[i]]])[0, 0]
                                    actual_original.append(original_val)
                                except Exception as e:
                                    print(f"Warning: Error inverse transforming actual {var}: {e}")
                                    actual_original.append(actual[i])
                            else:
                                actual_original.append(actual[i])
                        actuals_original.append(actual_original)
                    
                    # 4. BUFFER STRATEGY: Only keep the FIRST prediction (t+1)
                    if len(step_predictions_original) > 0 and len(actuals_original) > 0:
                        final_predictions.append(step_predictions_original[0])  # only t+1
                        final_actuals.append(actuals_original[0])
                        final_timestamps.append(df_online.index[t])

                    # 5. Inverse differencing with proper temporal alignment
                    # Use the corresponding row from the original forecasting data
                    forecast_idx = min(t - context_length, len(df_removed_nans_forecasting) - 1)
                    if forecast_idx >= 0:
                        last_actual_values = df_removed_nans_forecasting.iloc[forecast_idx][forecasting_variables].values
                    else:
                        # Fallback to mean values if index is out of bounds
                        last_actual_values = df_removed_nans_forecasting[forecasting_variables].mean().values

                    step_predictions_actual = self.inverse_difference(
                        step_predictions_original, 
                        last_actual_values
                    )

                    actuals_actual = self.inverse_difference(
                        actuals_original, 
                        last_actual_values
                    )

                    # 6. CLASSIFICATION with proper temporal alignment
                    if self.is_initialized and create_multihot_encoding is not None:
                        try:
                            classification_result = self._perform_classification(
                                step_predictions_actual, t, df_removed_nans_classification,
                                forecasting_variables
                            )
                            if classification_result:
                                print(f"Network Status: {classification_result}")
                        except Exception as e:
                            print(f"Warning: Classification failed at step {current_step}: {e}")

                    # Store actual predictions for plotting
                    if len(step_predictions_actual) > 0 and len(actuals_actual) > 0:
                        predictions_actuals.append(step_predictions_actual[0])  # Only t+1
                        actuals_actuals.append(actuals_actual[0])              # Only t+1

                    # 7. Send data to Dash plotter
                    if dash_plotter is not None and len(step_predictions_actual) > 0:
                        current_timestamp = df_online.index[t]
                        dash_plotter.add_buffer_predictions(
                            predictions=step_predictions_actual, 
                            actuals=actuals_actual, 
                            current_step=current_step,
                            current_datetime=current_timestamp,
                            variable_names=forecasting_variables
                        )
                    
                    # 8. Update context for next iteration
                    new_row = df_online_scaled.iloc[t].values.copy()
                    current_context = np.vstack((current_context[1:], new_row))
                    
                    # Progress reporting
                    if current_step % 50 == 0 or current_step < 5:
                        progress = (current_step * 100 / total_steps)
                        print(f"Step {current_step:3d}/{total_steps} ({progress:5.1f}%)")

                except Exception as e:
                    print(f"Error at step {current_step}: {e}")
                    # Continue with next iteration rather than failing completely
                    continue

            print("=" * 60)
            print("Rolling prediction completed!")

            # Create result DataFrames
            try:
                predictions_df = pd.DataFrame(
                    data=final_predictions,
                    columns=forecasting_variables,
                    index=pd.Index(range(context_length, context_length + len(final_predictions)), name='time_index')
                )

                actuals_df = pd.DataFrame(
                    data=final_actuals,
                    columns=forecasting_variables,
                    index=pd.Index(range(context_length, context_length + len(final_actuals)), name='time_index')
                )

                predictions_actuals_df = pd.DataFrame(
                    data=predictions_actuals,
                    columns=forecasting_variables,
                    index=pd.Index(range(context_length, context_length + len(predictions_actuals)), name='time_index')
                )

                actuals_actuals_df = pd.DataFrame(
                    data=actuals_actuals,
                    columns=forecasting_variables,
                    index=pd.Index(range(context_length, context_length + len(actuals_actuals)), name='time_index')
                )

                return predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df
                
            except Exception as e:
                print(f"Error creating result DataFrames: {e}")
                return None, None, None, None


# Convenience functions for backward compatibility
def predict_recursive_steps(initial_model, initial_context, variables, prediction_horizon=6, context_length=60):
    """Backward compatibility wrapper"""
    system = OnlineForecastingSystem()
    return system.predict_recursive_steps(initial_model, initial_context, variables, prediction_horizon, context_length)

def inverse_difference(predictions_arrays, last_actual_values):
    """Backward compatibility wrapper"""
    system = OnlineForecastingSystem()
    return system.inverse_difference(predictions_arrays, last_actual_values)

def rolling_buffer_prediction_with_dash(initial_model, df_online, scalers_train,
                                        context_length, df_removed_nans_forecasting, df_removed_nans_classification, 
                                        dash_plotter, classification_model_path=None, prediction_horizon=6, real_time_delay=0):
    """Backward compatibility wrapper with improved functionality"""
    if classification_model_path is None:
        system = OnlineForecastingSystem()
    else:
        system = OnlineForecastingSystem(models_dir=classification_model_path)
    
    return system.rolling_buffer_prediction_with_dash(
        initial_model, df_online, scalers_train, context_length,
        df_removed_nans_forecasting, df_removed_nans_classification, 
        dash_plotter, prediction_horizon, real_time_delay
    )

