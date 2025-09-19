from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import time
import sys
import os
import pickle
import matplotlib.pyplot as plt
import keyboard

# Ensure TensorFlow eager execution is enabled
tf.config.run_functions_eagerly(True)

# Global variables for classification model (loaded once)
_classification_model = None
_preprocessing_data = None

def load_classification_model(models_dir=None):
    global _classification_model, _preprocessing_data
    
    # Return cached model if already loaded
    if _classification_model is not None and _preprocessing_data is not None:
        print("Using cached classification model...")
        return (
            _classification_model,
            _preprocessing_data['label_to_index'],
            _preprocessing_data['index_to_label'],
            _preprocessing_data['label_to_name']
        )
    
    # Set default models directory
    if models_dir is None:
        models_dir = r"C:\ThesisWork\offical_approach\mth_project\mth_project\industrial_network_analysis\classification_model"
    
    try:
        print(f"Loading classification model from: {models_dir}")
        
        # Try to load improved model first, fallback to original
        model_candidates = [
            "improved_cnn_classifier.h5",
            "cnn_classifier.h5",
            "final_model.h5",
            "best_model.h5"
        ]
        
        model_path = None
        for model_name in model_candidates:
            candidate_path = os.path.join(models_dir, model_name)
            if os.path.exists(candidate_path):
                model_path = candidate_path
                print(f"Found model: {model_name}")
                break
        
        if model_path is None:
            raise FileNotFoundError(f"No classification model found in {models_dir}")
        
        # Load the model (using keras.models.load_model to avoid naming conflict)
        _classification_model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")
        
        # Try to load improved preprocessing first, fallback to original
        preprocessing_candidates = [
            "improved_preprocessing_objects.pkl",
            "preprocessing_objects.pkl",
            "encoders.pkl"
        ]
        
        preprocessing_path = None
        for preprocessing_name in preprocessing_candidates:
            candidate_path = os.path.join(models_dir, preprocessing_name)
            if os.path.exists(candidate_path):
                preprocessing_path = candidate_path
                print(f"Found preprocessing: {preprocessing_name}")
                break
        
        if preprocessing_path is None:
            raise FileNotFoundError(f"No preprocessing objects found in {models_dir}")
        
        # Load preprocessing objects
        with open(preprocessing_path, 'rb') as f:
            _preprocessing_data = pickle.load(f)
        print(f"Preprocessing objects loaded from: {preprocessing_path}")
        
        # Validate required keys
        required_keys = ['label_to_index', 'label_to_name']
        for key in required_keys:
            if key not in _preprocessing_data:
                raise KeyError(f"Missing required key '{key}' in preprocessing data")
        
        # Create index_to_label if not present
        if 'index_to_label' not in _preprocessing_data:
            _preprocessing_data['index_to_label'] = {
                v: k for k, v in _preprocessing_data['label_to_index'].items()
            }
            print("✓ Created index_to_label mapping")
        
        # print("✓ Classification model loaded successfully!")
        # print(f"  - Model input shape: {_classification_model.input_shape}")
        # print(f"  - Number of classes: {len(_preprocessing_data['label_to_index'])}")
        # print(f"  - Available classes: {list(_preprocessing_data['label_to_name'].keys())}")
        
        return (
            _classification_model,
            _preprocessing_data['label_to_index'],
            _preprocessing_data['index_to_label'],
            _preprocessing_data['label_to_name']
        )
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        raise
    except Exception as e:
        print(f"Error loading classification model: {e}")
        print(f"   Models directory: {models_dir}")
        print(f"   Directory exists: {os.path.exists(models_dir)}")
        if os.path.exists(models_dir):
            print(f"   Directory contents: {os.listdir(models_dir)}")
        raise

def get_classification_model_info():
    global _classification_model, _preprocessing_data
    
    if _classification_model is None or _preprocessing_data is None:
        return None
    
    return {
        'model_loaded': True,
        'input_shape': _classification_model.input_shape,
        'output_shape': _classification_model.output_shape,
        'num_classes': len(_preprocessing_data['label_to_index']),
        'class_names': list(_preprocessing_data['label_to_name'].keys()),
        'model_summary': _classification_model.summary
    }

def perform_classification(forecasted_actual, removed_nans_classification, t, classification_model):
    # forecasted_actual shape: (1, timesteps, forecasting_features)
    # We want: (timesteps, forecasting_features + classification_features)
    
    timesteps = forecasted_actual.shape[1]  # Should be 6
    forecasting_features = forecasted_actual.shape[2]  # Number of forecasting features
    
    # Extract the corresponding classification data for the same time window
    classification_start_idx = t
    classification_end_idx = t + timesteps
    
    # Get classification features for the same temporal window
    classification_data = removed_nans_classification.iloc[classification_start_idx:classification_end_idx].values
    
    # Reshape forecasted data: (1, timesteps, features) -> (timesteps, features)
    forecasted_reshaped = forecasted_actual.squeeze(0)  # Remove batch dimension
    
    # Concatenate along feature dimension: (timesteps, forecasting_features + classification_features)
    combined_input = np.concatenate([forecasted_reshaped, classification_data], axis=1)
    
    # Reshape for model input: (1, timesteps, total_features)
    model_input = combined_input.reshape(1, timesteps, -1)
    
    # print(f"Classification input shape: {model_input.shape}")
    # print(f"  - Timesteps: {timesteps}")
    # print(f"  - Forecasting features: {forecasting_features}")
    # print(f"  - Classification features: {classification_data.shape[1]}")
    # print(f"  - Total features: {model_input.shape[2]}")

    prediction = classification_model.predict(model_input)
    prediction_idx = np.argmax(prediction, axis=1)[0]
    prediction_confidence = prediction[0, prediction_idx]
    prediction_label = _preprocessing_data['index_to_label'][prediction_idx]
    prediction_name = _preprocessing_data['label_to_name'][prediction_label]

    return prediction_name

def predict_recursive_steps(initial_model, initial_context, variables, prediction_horizon=6, context_length=60):
    context = initial_context.copy()
    predictions = []
    
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

    return predictions

def improve_model(initial_model, training_data, target_data, epochs=1, batch_size=1):
    """
    Improve model with online learning using proper supervision.
    
    Args:
        initial_model: The model to improve
        training_data: Context window (batch_size, timesteps, features)
        target_data: Ground truth next step (batch_size, features)
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        improved_model: Updated model
    """
    try:
        # Calculate prediction error before training
        current_prediction = initial_model.predict(training_data, verbose=0)
        
        # Ensure target_data is numpy array
        if hasattr(target_data, 'numpy'):
            target_data_np = target_data.numpy()
        else:
            target_data_np = np.array(target_data)
            
        # Ensure prediction is numpy array
        if hasattr(current_prediction, 'numpy'):
            current_prediction_np = current_prediction.numpy()
        else:
            current_prediction_np = np.array(current_prediction)
            
        prediction_error = np.mean(np.abs(current_prediction_np - target_data_np))
        
        # Try training with existing optimizer first
        try:
            history = initial_model.fit(
                training_data, 
                target_data, 
                epochs=epochs, 
                batch_size=batch_size, 
                verbose=0
            )
        except ValueError as optimizer_error:
            if "Unknown variable" in str(optimizer_error):
                # Recompile with fresh optimizer if variable tracking issue
                print("   Recompiling model with fresh optimizer...")
                initial_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                    loss='mse', 
                    metrics=['mae']
                )
                # Try training again
                history = initial_model.fit(
                    training_data, 
                    target_data, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose=0
                )
            else:
                raise optimizer_error
        
        # Log improvement (simplified)
        if prediction_error > 0.01:  # Only log if error is significant
            print(f"   Model training completed, pre-training error: {prediction_error:.4f}")
            
    except Exception as e:
        print(f"   Warning: Model training failed ({e})")
        print("   Continuing with original model...")
        # If all training attempts fail, return the original model unchanged
    
    return initial_model

### good inverse transform function for differenced data
def inverse_difference(predictions_arrays, last_actual_values):
    actual_predictions = []
    current_values = last_actual_values.copy()
    
    for pred_diff in predictions_arrays:
        # Add difference to get actual value
        current_values = current_values + np.array(pred_diff)
        actual_predictions.append(current_values.copy())
    
    return actual_predictions

# learning version is alternative that improves the model during each step 

def rolling_buffer_learning_prediction_with_dash(initial_model, 
                                        df_online, 
                                        scalers, 
                                        context_length,
                                        df_removed_nans_forecasting, 
                                        df_removed_nans_classification,
                                        dash_plotter, 
                                        variables, 
                                        prediction_horizon=6, 
                                        classification_model_path=None):
    # Prepare model for online learning by recompiling with fresh optimizer
    print("Preparing model for online learning...")
    try:
        initial_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001), 
            loss='mse', 
            metrics=['mae']
        )
        print("Model successfully prepared for online learning")
    except Exception as e:
        print(f"Warning: Could not recompile model ({e}), will try per-step recompilation")
    
    # Scale the online data using the same scalers from training
    scaled_data = np.zeros_like(df_online.values)
    for i, var in enumerate(variables):
        scaler = scalers[var]
        scaled_data[:, i] = scaler.transform(df_online[var].values.reshape(-1, 1)).flatten()
    
    final_predictions = []
    final_actuals = []
    final_timestamps = []

    predictions_actuals = []
    actuals_actuals = []
    total_steps = len(scaled_data) - context_length - prediction_horizon + 1

    if dash_plotter is not None:
        dash_plotter.set_total_steps(total_steps)

    current_context = scaled_data[:context_length].copy() 
    model = initial_model

    # Load classification model once at the beginning
    try:
        if classification_model_path is None:
            classification_model_path = r"C:\ThesisWork\offical_approach\mth_project\mth_project\industrial_network_analysis\classification_model"
        
        classification_model, label_to_index, index_to_label, label_to_name = load_classification_model(classification_model_path)
        classification_enabled = True
        print("Classification model initialized successfully")
    except Exception as e:
        print(f"Warning: Could not load classification model: {e}")
        print("   Continuing with forecasting only...")
        classification_enabled = False
        classification_model = None

    # Main prediction loop
    for t in range(context_length, len(scaled_data) - prediction_horizon + 1):
        # wait 30 seconds (for demo - real data is 1 minute apart)
        time.sleep(30)
        current_step = t - context_length

        # 1. Make predictions for next 'prediction_horizon' steps
        step_predictions = predict_recursive_steps(model, current_context, variables = variables, prediction_horizon = prediction_horizon)
        print(f"Step {t}, raw predictions: {step_predictions}")
        
        # 2. Convert all predictions to original scale
        step_predictions_original = []
        for pred in step_predictions:
            pred_original = []
            for i, var in enumerate(variables):
                # Filter status columns
                if 'status' in var.lower():
                    pred[i] = np.round(pred[i])
                
                scaler = scalers[var]
                original_val = scaler.inverse_transform([[pred[i]]])[0, 0]
                pred_original.append(original_val)
            step_predictions_original.append(pred_original)
        
        print(f"Step {t}, inverse transformed predictions: {step_predictions_original}")

        # 3. Get actual values for all predicted steps
        actual_values = []
        for step in range(prediction_horizon):
            if t + step < len(scaled_data):
                actual_values.append(scaled_data[t + step, :])
        
        actuals_original = []
        for actual in actual_values:
            actual_original = []
            for i, var in enumerate(variables):
                scaler = scalers[var]
                original_val = scaler.inverse_transform([[actual[i]]])[0, 0]
                actual_original.append(original_val)
            actuals_original.append(actual_original)
        
        # 4. BUFFER STRATEGY: Only keep the FIRST prediction (t+1)
        if len(step_predictions_original) > 0 and len(actuals_original) > 0:
            final_predictions.append(step_predictions_original[0])  # only t+1
            final_actuals.append(actuals_original[0])
            final_timestamps.append(df_online.index[t])

        # 5. Inverse differencing for plotting
        last_actual_index = t - context_length
        if last_actual_index >= 0 and last_actual_index < len(df_removed_nans_forecasting):
            last_actual_values = df_removed_nans_forecasting.iloc[last_actual_index][variables].values
        else:
            last_actual_values = df_removed_nans_forecasting[variables].mean().values

        step_predictions_actual = inverse_difference(
            step_predictions_original, 
            last_actual_values
        )

        actuals_actual = inverse_difference(
            actuals_original, 
            last_actual_values
        )

        print(f"Step {t}, inverse differenced predictions all: {step_predictions_actual}")
        print(f"Step {t}, inverse differenced predictions (t+1): {step_predictions_actual[0]}")
        print(f"Step {t}, inverse differenced actuals: {actuals_actual}")

        # 6. Classification (Only if model is loaded), with proper temporal alignment
        classification_result = None
        if classification_enabled and classification_model is not None:
            try:
                timesteps = prediction_horizon
                features = len(variables)

                step_predictions_actual_array = np.array(step_predictions_actual)
                
                step_predictions_actual_classification = step_predictions_actual_array.reshape(1, timesteps, features)
                
                # Make prediction
                classification_result = perform_classification(step_predictions_actual_classification, df_removed_nans_classification, t, classification_model)
                print(f"🔍 Classification prediction: {classification_result}")
                
            except Exception as e:
                print(f"⚠️ Classification error: {e}")
                classification_result = "Classification Error"

        if len(step_predictions_actual) > 0 and len(actuals_actual) > 0:
            predictions_actuals.append(step_predictions_actual[0])  # Only t+1
            actuals_actuals.append(actuals_actual[0])              # Only t+1

        # 7. Send data to Dash plotter (all buffer predictions)
        if dash_plotter is not None:
            if len(step_predictions_actual) > 0:
                # print("DEBUG: About to call add_buffer_predictions")
                current_timestamp = df_online.index[t]
                
                # Pass the ACTUAL prediction for t+1 (already computed!)
                # step_predictions_actual[0] is the prediction for t+1
                saved_prediction_t1 = step_predictions_actual[0] if len(step_predictions_actual) > 0 else None
                
                # For the red line extension, we use the SAME prediction for t+1
                # This IS the future prediction - no need to recompute!
                future_prediction_t1 = step_predictions_actual[0] if len(step_predictions_actual) > 0 else None
                
                try:
                    dash_plotter.add_buffer_predictions(
                        predictions=step_predictions_actual, 
                        actuals=actuals_actual, 
                        current_step=current_step,
                        current_datetime=current_timestamp,
                        variable_names=variables,
                        saved_prediction=saved_prediction_t1,
                        future_prediction=future_prediction_t1  # Same as saved prediction - it IS the future!
                    )
                    
                    # Add classification result if available
                    # if classification_result is not None:
                    #     dash_plotter.add_classification_result(classification_result)
                    
                    # print("DEBUG: add_buffer_predictions called successfully")
                except Exception as e:
                    print(f"DEBUG: Error in add_buffer_predictions: {e}")
            else:
                print("DEBUG: step_predictions_actual is empty, not calling dash plotter")
        else:
            print("DEBUG: dash_plotter is None, not calling dash plotter")

        # 8. Update context and model for next iteration
        new_row = scaled_data[t, :].copy()
        
        # Prepare training data for model improvement
        # Train to predict next step (t+1) using context up to current step (t)
        if t + 1 < len(scaled_data):  # Ensure we have ground truth for t+1
            # Context: up to current step t
            training_context = np.vstack((current_context[1:], new_row))
            # Target: next step t+1 (ground truth)
            target_next_step = scaled_data[t + 1, :].copy()
            
            # Train model to predict t+1 from context ending at t
            model = improve_model(
                model, 
                training_context.reshape(1, context_length, len(variables)), 
                target_next_step.reshape(1, len(variables)), 
                epochs=1, 
                batch_size=1
            )
            print(f"Model improved at step {current_step}: training to predict t+1")
        
        # Update context for next iteration
        current_context = np.vstack((current_context[1:], new_row))
        
        # 9. if 'Q' is pressed by user, end program and clean all variables
        if keyboard.is_pressed('q'):
            print("Exiting program...")
            # Clean up variables
            del df_online
            del scaled_data
            del current_context
            del final_predictions
            del final_actuals
            del final_timestamps
            del step_predictions
            del actual_values
            del scalers_train
            del context_length
            del df_removed_nans_forecasting
            del df_removed_nans_classification
            del variables
            break

    print("=" * 60)
    print("Rolling prediction completed!")

    predictions_df = pd.DataFrame(
        data=final_predictions,
        columns=variables,
        index=pd.Index(range(context_length, context_length + len(final_predictions)), name='time_index')
    )

    actuals_df = pd.DataFrame(
        data=final_actuals,
        columns=variables,
        index=pd.Index(range(context_length, context_length + len(final_actuals)), name='time_index')
    )

    predictions_actuals_df = pd.DataFrame(
        data=predictions_actuals,
        columns=variables,
        index=pd.Index(range(context_length, context_length + len(predictions_actuals)), name='time_index')
    )

    actuals_actuals_df = pd.DataFrame(
        data=actuals_actuals,
        columns=variables,
        index=pd.Index(range(context_length, context_length + len(actuals_actuals)), name='time_index')
    )

    return predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df