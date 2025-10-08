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

def predict_recursive_steps(model, initial_context, variables, prediction_horizon=6, context_length=60):
    context = initial_context.copy()
    predictions = []
    
    for step in range(prediction_horizon):
        # Reshape for model input (batch_size=1, timesteps, features)
        context_reshaped = context.reshape(1, context_length, len(variables))
        
        # Make prediction for next step
        step_prediction = model.predict(context_reshaped, verbose=0)
        
        # Store the prediction (flattened for compatibility)
        predictions.append(step_prediction.flatten())
        
        # This is the predicted next timestep
        new_row = step_prediction[0].copy()  # Extract from batch dimension and copy
        
        # Slide context window: remove oldest, add new prediction
        context = np.vstack((context[1:], new_row.reshape(1, -1)))
    
    return predictions

def improve_model(initial_model, training_data, target_data, prediction_error_mae):
    try:
        print(f"   Attempting model improvement (current MAE: {prediction_error_mae:.6f})")
        
        # Store original weights for potential rollback
        original_weights = initial_model.get_weights()
        
        # Convert target data to numpy for analysis
        if hasattr(target_data, 'numpy'):
            target_data_np = target_data.numpy()
        else:
            target_data_np = np.array(target_data)

        # Fixed hyperparameters
        learning_rate = 0.001
        epochs = 5
        batch_size = 32
        
        print(f"   → Using fixed LR: {learning_rate}, epochs: {epochs}, batch_size: {batch_size}")
        
        # Compile model with fixed learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        initial_model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        # Train the model (no sample weights)
        try:
            time_start = time.time()
            
            history = initial_model.fit(
                training_data, 
                target_data, 
                epochs=epochs, 
                batch_size=batch_size,
                verbose=0
            )
            
            # Simple improvement check - just compare global MAE
            new_prediction = initial_model.predict(training_data, verbose=0)
            if hasattr(new_prediction, 'numpy'):
                new_prediction_np = new_prediction.numpy()
            else:
                new_prediction_np = np.array(new_prediction)
            
            # Calculate new global error
            new_global_error = np.mean(np.abs(new_prediction_np - target_data_np))
            
            time_stop = time.time()    
            print(f"   Model training took {time_stop - time_start:.2f} seconds")  
            
            # Simple decision: accept if global error improved
            if new_global_error < prediction_error_mae:
                improvement_ratio = (prediction_error_mae - new_global_error) / (prediction_error_mae + 1e-8)
                print(f"   ✓ Model improved: {prediction_error_mae:.6f} → {new_global_error:.6f} (Δ{improvement_ratio*100:.1f}%)")
                return initial_model
            else:
                # Revert to original weights if no improvement
                initial_model.set_weights(original_weights)
                print(f"   ✗ No improvement: {prediction_error_mae:.6f} → {new_global_error:.6f}, reverting weights")
                return initial_model

        except ValueError as optimizer_error:
            print(f"   Skipping training: optimizer error ({optimizer_error})")
            return initial_model

    except Exception as e:
        print(f"   Warning: Model training failed ({e})")
        print("   Continuing with original model...")
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

def one_step_rolling_buffer_learning_prediction_with_dash(initial_model, 
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
    print("\n Preparing model for online learning...")
    try:
        initial_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001), 
            loss='mse', 
            metrics=['mae']
        )
        print("\n Model successfully prepared for online learning")
    except Exception as e:
        print(f"Warning: Could not recompile model ({e}), will try per-step recompilation")
    
    # FIX ISSUE 1: Models expect raw data and do their own scaling - don't double scale!
    # The models were trained on raw data and have StandardScaler built into their training process.
    # Adding another scaling step here would result in double scaling: Raw → Scale → Scale → Model
    # Instead, we use raw data directly: Raw → Model (which handles scaling internally)
    print("   Using raw data (models handle scaling internally)")
    scaled_data = df_online.values  # Keep variable name for compatibility, but no scaling applied
    
    final_predictions = []
    final_actuals = []
    final_timestamps = []

    predictions_actuals = []
    actuals_actuals = []
    total_steps = len(scaled_data) - context_length - prediction_horizon + 1

    if dash_plotter is not None:
        dash_plotter.set_total_steps(total_steps)

    current_context = scaled_data[:context_length].copy()  # Raw data context
    model = initial_model
    
    # Store previous predictions for learning from historical errors
    previous_prediction = None
    previous_context = None

    # Load classification model once at the beginning
    try:
        if classification_model_path is None:
            classification_model_path = r"C:\ThesisWork\offical_approach\mth_project\mth_project\industrial_network_analysis\classification_model"
        
        classification_model, label_to_index, index_to_label, label_to_name = load_classification_model(classification_model_path)
        classification_enabled = True
        
        # Set the label_to_name dictionary in the dashboard
        if dash_plotter is not None:
            dash_plotter.set_label_to_name_dict(label_to_name)

    except Exception as e:
        print(f"Warning: Could not load classification model: {e}")
        print("   Continuing with forecasting only...")
        classification_enabled = False
        classification_model = None


    # Main prediction loop
    for t in range(context_length, len(scaled_data) - prediction_horizon + 1):
        # wait 1 seconds (for demo - real data is 1 minute apart)
        time.sleep(1)
        current_step = t - context_length

        # 1. Make predictions for next 'prediction_horizon' steps using selected method
        try:
            step_predictions = predict_recursive_steps(model, current_context, variables=variables, prediction_horizon=prediction_horizon)    
        except Exception as e:
            print(f" Prediction failed ({e}).")
                    
        # 2. Model outputs are already in original scale - no inverse transform needed
        step_predictions_original = []
        for pred in step_predictions:
            pred_original = []
            for i, var in enumerate(variables):
                # Filter status columns
                if 'status' in var.lower():
                    pred[i] = np.round(pred[i])
                
                # No scaling conversion needed - model outputs original scale
                original_val = pred[i]
                pred_original.append(original_val)
            step_predictions_original.append(pred_original)
        
        # 3. Get actual values for all predicted steps (already in original scale)
        actual_values = []
        for step in range(prediction_horizon):
            if t + step < len(scaled_data):
                actual_values.append(scaled_data[t + step, :])
        
        actuals_original = []
        for actual in actual_values:
            actual_original = []
            for i, var in enumerate(variables):
                # No scaling conversion needed - data is already in original scale
                original_val = actual[i]
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
 
        # 6. Classification (Only if model is loaded), with proper temporal alignment
        if classification_enabled and classification_model is not None:
            try:
                # Create classification input using historical context + first prediction
                # Take last 5 timesteps from context + first prediction = 6 timesteps total
                historical_part = current_context[-5:]  # Shape: (5, features)
                
                if len(step_predictions_original) > 0:
                    # Add the first prediction as the 6th timestep
                    first_prediction = np.array(step_predictions_original[0]).reshape(1, -1)  # Shape: (1, features)
                    classification_input = np.vstack([historical_part, first_prediction])  # Shape: (6, features)
                else:
                    # Fallback: use last 6 timesteps from context
                    classification_input = current_context[-6:]  # Shape: (6, features)
                
                # Reshape for model input: (batch_size=1, timesteps=6, features)
                classification_input_reshaped = classification_input.reshape(1, 6, len(variables))
                
                # Perform classification
                classification_result = classification_model.predict(classification_input_reshaped, verbose=0)
                result = np.argmax(classification_result, axis=1)
                result_to_label = index_to_label[result[0]]

                classification_result_name = label_to_name[result_to_label]
                print(f"Classification result: {classification_result_name}")

            except Exception as e:
                print(f"Classification error: {e}")
                classification_result = None
                result = None
        else:
            classification_result = None
            result = None

        port_statuses_check = df_removed_nans_classification.iloc[t]
        port_statuses = {}
        for name, status in port_statuses_check.items():
            if status not in [None, np.nan]:
                port_statuses[name] = status
        
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
                    # Prepare classification result if available
                    classification_result_data = None
                    if classification_enabled and 'classification_result_name' in locals() and 'classification_result' in locals():
                        classification_result_data = {
                            'classification': classification_result_name,
                            'confidence': float(np.max(classification_result))
                        }
                    
                    dash_plotter.add_buffer_predictions(
                        predictions=step_predictions_actual, 
                        actuals=actuals_actual, 
                        current_step=current_step,
                        current_datetime=current_timestamp,
                        variable_names=variables,
                        saved_prediction=saved_prediction_t1,
                        future_prediction=future_prediction_t1,
                        port_statuses=port_statuses if len(port_statuses) > 0 else None,
                        classification_result=classification_result_data
                    )
                    
                    # Add classification result if available
                    if classification_result is not None:
                        dash_plotter.add_classification_result(classification_result)
                    
                    print("DEBUG: add_buffer_predictions called successfully")
                except Exception as e:
                    print(f"DEBUG: Error in add_buffer_predictions: {e}")
            else:
                print("DEBUG: step_predictions_actual is empty, not calling dash plotter")
        else:
            print("DEBUG: dash_plotter is None, not calling dash plotter")

        # 8. Update context and model for next iteration
        new_row = scaled_data[t, :].copy()  # Raw data row
        
        # learn from historical prediction error (simplified for raw data)
        if previous_prediction is not None and previous_context is not None:
            # The actual value that just arrived (time t) - raw data
            actual_current_step = scaled_data[t, :].copy()
            
            # Calculate the error from our previous prediction (raw scale comparison)
            historical_prediction_error = np.mean(np.abs(previous_prediction.flatten() - actual_current_step))
            
            print(f"   Learning from historical error at step {current_step}: MAE = {historical_prediction_error:.6f}")
            
            # Train the model on the historical prediction error
            # Use the context that was available when we made the previous prediction
            model = improve_model(
                model, 
                previous_context.reshape(1, context_length, len(variables)), 
                actual_current_step.reshape(1, len(variables)), 
                prediction_error_mae=historical_prediction_error
            )
        
        # Prepare context for the NEXT prediction (t+1)
        updated_context = np.vstack((current_context[1:], new_row))
        
        # Make prediction for the NEXT step (t+1) - this will be evaluated in the next iteration
        if t + 1 < len(scaled_data):
            next_prediction = model.predict(
                updated_context.reshape(1, context_length, len(variables)), 
                verbose=0
            )
            
            # Store this prediction and context for learning in the next iteration
            previous_prediction = next_prediction.copy()
            previous_context = updated_context.copy()
        
        # Update context for next iteration
        current_context = updated_context
        
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

