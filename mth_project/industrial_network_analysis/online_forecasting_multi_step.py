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
import threading

# Ensure TensorFlow eager execution is enabled
tf.config.run_functions_eagerly(True)

# Global stop flag for graceful shutdown
stop_flag = threading.Event()

# Global variables for classification model (loaded once)
_classification_model = None
_preprocessing_data = None

def keyboard_listener():
    """Run in separate thread to listen for quit command"""
    while not stop_flag.is_set():
        try:
            if keyboard.is_pressed('q'):
                print("\n🛑 Quit command detected! Initiating graceful shutdown...")
                stop_flag.set()
                break
        except Exception as e:
            # Keyboard library might fail in some environments
            pass
        time.sleep(0.1)  # Check every 100ms

def interruptible_sleep(duration, check_interval=0.1):
    """Sleep that can be interrupted by stop_flag"""
    end_time = time.time() + duration
    while time.time() < end_time:
        if stop_flag.is_set():
            break
        time.sleep(min(check_interval, end_time - time.time()))

def cleanup_and_create_results(final_predictions, final_actuals, final_timestamps, 
                              predictions_actuals, actuals_actuals, variables, 
                              context_length, current_step):
    """Clean shutdown procedure with result saving"""
    print("🧹 Performing graceful cleanup...")
    
    try:
        # Create DataFrames from collected data
        if len(final_predictions) > 0:
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
            
            print(f"✅ Successfully created results for {len(final_predictions)} predictions")
        else:
            # Create empty DataFrames if no predictions were made
            predictions_df = pd.DataFrame(columns=variables)
            actuals_df = pd.DataFrame(columns=variables)
            print("⚠️  No predictions were completed before shutdown")
        
        # Create actuals DataFrames if available
        if len(predictions_actuals) > 0:
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
        else:
            predictions_actuals_df = pd.DataFrame(columns=variables)
            actuals_actuals_df = pd.DataFrame(columns=variables)
        
        print(f"📊 Results summary:")
        print(f"   - Completed steps: {current_step}")
        print(f"   - Predictions collected: {len(final_predictions)}")
        print(f"   - Variables tracked: {len(variables)}")
        
        return predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        # Return empty DataFrames on error
        empty_df = pd.DataFrame(columns=variables)
        return empty_df, empty_df, empty_df, empty_df

def clear_large_variables(*var_names):
    """Clear large variables to free memory"""
    import gc
    cleared_count = 0
    frame = sys._getframe(1)  # Get caller's frame
    
    for var_name in var_names:
        if var_name in frame.f_locals:
            try:
                del frame.f_locals[var_name]
                cleared_count += 1
            except:
                pass
    
    gc.collect()  # Force garbage collection
    if cleared_count > 0:
        print(f"🗑️  Cleared {cleared_count} large variables from memory")

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

def predict_multistep_direct(model, context, variables, prediction_horizon=6):
    """
    Multi-step direct prediction - model outputs all future steps at once.
    """
    # Reshape context for model input
    context_reshaped = context.reshape(1, context.shape[0], len(variables))
    
    # Get multi-step prediction (flattened output)
    multistep_pred = model.predict(context_reshaped, verbose=0)
    
    # Reshape output to separate timesteps
    n_features = len(variables)
    predictions = []
    
    for step in range(prediction_horizon):
        start_idx = step * n_features
        end_idx = (step + 1) * n_features
        step_pred = multistep_pred[0, start_idx:end_idx]
        predictions.append(step_pred)
    
    return predictions

def improve_model(initial_model, training_data, target_data, prediction_error_mae):
    """
    Simplified model improvement function for multi-step models.
    Adapted to handle multi-step target data.
    """
    
    try:
        print(f"   Attempting multi-step model improvement (current MAE: {prediction_error_mae:.6f})")
        
        # Store original weights for potential rollback
        original_weights = initial_model.get_weights()
        
        # Convert target data to numpy for analysis
        if hasattr(target_data, 'numpy'):
            target_data_np = target_data.numpy()
        else:
            target_data_np = np.array(target_data)
        
        # Simple fixed hyperparameters
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
            print(f"   Multi-step model training took {time_stop - time_start:.2f} seconds")  
            
            # Simple decision: accept if global error improved
            if new_global_error < prediction_error_mae:
                improvement_ratio = (prediction_error_mae - new_global_error) / (prediction_error_mae + 1e-8)
                print(f"   ✓ Multi-step model improved: {prediction_error_mae:.6f} → {new_global_error:.6f} (Δ{improvement_ratio*100:.1f}%)")
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
        print(f"   Warning: Multi-step model training failed ({e})")
        print("   Continuing with original model...")
        return initial_model

def inverse_difference(predictions_arrays, last_actual_values):
    """Convert differenced predictions back to actual values."""
    actual_predictions = []
    current_values = last_actual_values.copy()
    
    for pred_diff in predictions_arrays:
        # Add difference to get actual value
        current_values = current_values + np.array(pred_diff)
        actual_predictions.append(current_values.copy())
    
    return actual_predictions

def multistep_rolling_buffer_learning_prediction_with_dash(initial_model, 
                                        df_online, 
                                        scalers, 
                                        context_length,
                                        df_removed_nans_forecasting, 
                                        df_removed_nans_classification,
                                        dash_plotter, 
                                        variables, 
                                        prediction_horizon=6, 
                                        classification_model_path=None):

    # Initialize graceful shutdown system
    global stop_flag
    stop_flag.clear()  # Reset the flag for this run
    
    # Start keyboard listener in separate thread
    print("⌨️  Starting keyboard listener (Press 'q' to quit gracefully)...")
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    # Prepare model for online learning by recompiling with fresh optimizer
    print("\n Preparing multi-step model for online learning...")
    try:
        initial_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001), 
            loss='mse', 
            metrics=['mae']
        )
        print("\n Multi-step model successfully prepared for online learning")
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
        print("   Continuing with multi-step forecasting only...")
        classification_enabled = False
        classification_model = None

    # Main prediction loop
    for t in range(context_length, len(scaled_data) - prediction_horizon + 1):
        # Check for quit signal at the start of each iteration
        if stop_flag.is_set():
            print("🛑 Graceful shutdown initiated...")
            break
            
        current_step = t - context_length
        
        # Interruptible sleep (can be interrupted by quit signal)
        interruptible_sleep(1)  # wait 1 second (for demo)
        
        # Check again after sleep in case quit was requested
        if stop_flag.is_set():
            print("🛑 Graceful shutdown initiated...")
            break

        # 1. Make multi-step predictions using direct method
        try:
            step_predictions = predict_multistep_direct(model, current_context, variables=variables, prediction_horizon=prediction_horizon)    
        except Exception as e:
            print(f" Multi-step prediction failed ({e}).")
                    
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
        
        # 4. BUFFER STRATEGY: Only keep the FIRST prediction (t+1) for evaluation
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
                current_timestamp = df_online.index[t]
                
                # Pass the ACTUAL prediction for t+1 (already computed!)
                saved_prediction_t1 = step_predictions_actual[0] if len(step_predictions_actual) > 0 else None
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

        # 8. Update context and model for next iteration - MULTI-STEP ADAPTATION
        new_row = scaled_data[t, :].copy()
        
        # REALISTIC ONLINE LEARNING: Learn from historical prediction error
        if previous_prediction is not None and previous_context is not None:
            # The actual values that just arrived (multiple steps)
            actual_current_steps = []
            num_previous_steps = len(previous_prediction)
            
            for step in range(num_previous_steps):
                if t + step < len(scaled_data):
                    actual_current_steps.append(scaled_data[t + step, :].copy())
            
            if len(actual_current_steps) > 0:
                # Flatten previous multi-step prediction for comparison
                previous_pred_flattened = np.concatenate(previous_prediction, axis=0)
                actual_steps_flattened = np.concatenate(actual_current_steps, axis=0)
                
                # Calculate the error from our previous multi-step prediction
                historical_prediction_error = np.mean(np.abs(previous_pred_flattened - actual_steps_flattened))
                
                print(f"   Learning from multi-step historical error at step {current_step}: MAE = {historical_prediction_error:.6f}")
                
                # Create multi-step target for training
                multistep_target = actual_steps_flattened.reshape(1, -1)
                
                # Train the model on the historical multi-step prediction error
                model = improve_model(
                    model, 
                    previous_context.reshape(1, context_length, len(variables)), 
                    multistep_target, 
                    prediction_error_mae=historical_prediction_error
                )
        
        # Prepare context for the NEXT multi-step prediction
        updated_context = np.vstack((current_context[1:], new_row))
        
        # Make multi-step prediction for the NEXT steps - this will be evaluated in the next iteration
        if t + prediction_horizon < len(scaled_data):
            next_predictions = predict_multistep_direct(model, updated_context, variables, prediction_horizon)
            
            # Store this prediction and context for learning in the next iteration
            previous_prediction = [pred.copy() for pred in next_predictions]
            previous_context = updated_context.copy()
        
        # Update context for next iteration
        current_context = updated_context

    # End of main loop - either completed naturally or stopped by user
    print("=" * 60)
    
    if stop_flag.is_set():
        print("Multi-step rolling prediction stopped by user request!")
        print(f"📈 Processed {current_step + 1} steps before stopping")
    else:
        print("Multi-step rolling prediction completed successfully!")
        print(f"📈 Processed all {current_step + 1} steps")
    
    # Create results using graceful cleanup function
    try:
        predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df = cleanup_and_create_results(
            final_predictions, final_actuals, final_timestamps,
            predictions_actuals, actuals_actuals, variables,
            context_length, current_step
        )
        
        # Clear large variables to free memory
        clear_large_variables('df_online', 'scaled_data', 'current_context', 
                            'step_predictions', 'actual_values', 'model',
                            'initial_model', 'classification_model')
        
        print("✅ Graceful cleanup completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during final cleanup: {e}")
        # Return empty DataFrames as fallback
        empty_df = pd.DataFrame(columns=variables)
        predictions_df = actuals_df = predictions_actuals_df = actuals_actuals_df = empty_df

    return predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df
