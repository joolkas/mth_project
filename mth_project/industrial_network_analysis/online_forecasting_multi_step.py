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
        
        # print(f"📊 Results summary:")
        # print(f"   - Completed steps: {current_step}")
        # print(f"   - Predictions collected: {len(final_predictions)}")
        # print(f"   - Variables tracked: {len(variables)}")
        
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
    """
    Simplified multi-step forecasting with Dash integration
    Removes complex features for production stability
    """

    # Initialize graceful shutdown system (simplified for production)
    global stop_flag
    stop_flag.clear()  # Reset the flag for this run
    
    print("🚀 Starting multi-step forecasting pipeline...")
    print(f"📊 Data info: {len(df_online)} samples, {len(variables)} variables, context_length={context_length}")
    print(f"🎯 Variables: {variables}")
    print(f"📅 Time range: {df_online.index[0]} to {df_online.index[-1]}")
    
    # Skip keyboard listener for production stability
    
    # Prepare model for online learning (simplified)
    # print("🔧 Preparing multi-step model...")
    model = initial_model  # Use model as-is for stability
    
    # MAJOR FIX: Apply differencing FIRST, then scaling (same as training)
    # Models were trained on: scaled(differenced(data))
    print("   Applying differencing to online data (same as training)")
    df_online_differenced = df_online.diff().dropna()
    
    print("   Scaling differenced online data with training scalers")
    scaled_data = np.zeros_like(df_online_differenced.values)
    
    scaling_errors = []
    for i, var in enumerate(variables):
        if var in scalers:
            scaler = scalers[var]
            try:
                scaled_data[:, i] = scaler.transform(df_online_differenced[var].values.reshape(-1, 1)).flatten()
            except Exception as e:
                scaling_errors.append(f"{var}: {e}")
                # Fallback: use standardization
                data_values = df_online_differenced[var].values
                scaled_data[:, i] = (data_values - data_values.mean()) / (data_values.std() + 1e-8)
        else:
            scaling_errors.append(f"{var}: scaler not found")
            # Fallback: use standardization
            data_values = df_online_differenced[var].values
            scaled_data[:, i] = (data_values - data_values.mean()) / (data_values.std() + 1e-8)
    
    if scaling_errors:
        print(f"⚠️  Scaling issues: {len(scaling_errors)} variables had problems")
        for error in scaling_errors[:3]:  # Show first 3 errors
            print(f"   • {error}")
    
    final_predictions = []
    final_actuals = []
    final_timestamps = []

    predictions_actuals = []
    actuals_actuals = []
    total_steps = len(scaled_data) - context_length - prediction_horizon + 1
    print(f"   After differencing: {len(df_online)} → {len(df_online_differenced)} samples")
    print(f"   After scaling: {scaled_data.shape}")
    print(f"   Processing {total_steps} prediction steps")

    # print(f"📈 Will process {total_steps} prediction steps")
    
    if dash_plotter is not None:
        try:
            dash_plotter.set_total_steps(total_steps)
            # print("🎯 Dashboard configured successfully")
        except Exception as e:
            print(f"⚠️  Dashboard configuration failed: {e}")
            dash_plotter = None

    current_context = scaled_data[:context_length].copy()

    # print(f"🔄 Initial context shape: {current_context.shape}")
    
    # Track processing time
    start_time = time.time()

    # Disable classification for production stability (can be re-enabled later)
    print("🔧 Classification disabled for production stability")
    classification_enabled = False
    classification_model = None

    # Main prediction loop
    for t in range(context_length, len(scaled_data) - prediction_horizon + 1):
        current_step = t - context_length
        
        # Show progress every 10 steps
        if current_step % 10 == 0:
            progress_pct = (current_step / total_steps) * 100
            #print(f"📊 Progress: {current_step}/{total_steps} steps ({progress_pct:.1f}%)")
        
        # No sleep in production mode for faster processing

        # 1. Make multi-step predictions using direct method
        try:
            step_predictions = predict_multistep_direct(model, current_context, variables=variables, prediction_horizon=prediction_horizon)    
        except Exception as e:
            print(f"⚠️  Multi-step prediction failed ({e}), skipping this step")
            continue  # Skip this iteration if prediction fails
                    
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
        
        # 3. Get actual values for all predicted steps (FIXED: t+1 to t+6, not t to t+5)
        actual_values = []
        for step in range(prediction_horizon):
            actual_index = t + 1 + step  # FIXED: Start from t+1, not t
            if actual_index < len(scaled_data):
                actual_values.append(scaled_data[actual_index, :])
        
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
            # FIXED: Use correct timestamp for t+1 prediction (from differenced data)
            if t + 1 < len(df_online_differenced):
                final_timestamps.append(df_online_differenced.index[t + 1])
            else:
                final_timestamps.append(df_online_differenced.index[t])

        # 5. Inverse differencing for plotting 
        # MAJOR FIX: Use original df_online (non-differenced) as base for inverse differencing
        # The differenced index t corresponds to original index t+1 (since diff() drops first row)
        original_base_index = t  # This maps to the original data before differencing
        if original_base_index < len(df_online):
            last_actual_values = df_online.iloc[original_base_index][variables].values
        else:
            # Fallback
            last_actual_values = df_online.iloc[-1][variables].values

        step_predictions_actual = inverse_difference(
            step_predictions_original, 
            last_actual_values
        )

        actuals_actual = inverse_difference(
            actuals_original, 
            last_actual_values
        )

        # 6. Classification (Disabled for production stability)
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
                # Use correct timestamp from differenced data for dashboard context
                current_timestamp = df_online_differenced.index[t]
                
                # Pass the ACTUAL prediction for t+1 (already computed!)
                saved_prediction_t1 = step_predictions_actual[0] if len(step_predictions_actual) > 0 else None
                future_prediction_t1 = step_predictions_actual[0] if len(step_predictions_actual) > 0 else None
                
                try:
                    # Classification disabled for stability
                    classification_result_data = None
                    
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
                    
                    # print("DEBUG: add_buffer_predictions called successfully")
                except Exception as e:
                    print(f"DEBUG: Error in add_buffer_predictions: {e}")
            else:
                print("DEBUG: step_predictions_actual is empty, not calling dash plotter")
        else:
            print("DEBUG: dash_plotter is None, not calling dash plotter")

        # 8. Update context for next iteration (simplified - no online learning)
        new_row = scaled_data[t, :].copy()
        current_context = np.vstack((current_context[1:], new_row))
        
        # Calculate and display prediction error for monitoring
        if len(step_predictions_original) > 0 and len(actuals_original) > 0:
            prediction_error = np.mean(np.abs(np.array(step_predictions_original[0]) - np.array(actuals_original[0])))
            if current_step % 20 == 0:  # Show error every 20 steps
                print(f"   📉 Step {current_step} prediction error (MAE): {prediction_error:.6f}")

    # End of main loop
    print("=" * 60)
    print("✅ Multi-step rolling prediction completed successfully!")
    print(f"📈 Processed all {current_step + 1} steps")
    print(f"🎯 Generated {len(final_predictions)} predictions")
    processing_time = time.time() - start_time
    print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
    print(f"⚡ Average time per step: {processing_time/max(1, current_step+1):.3f} seconds")
    
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
