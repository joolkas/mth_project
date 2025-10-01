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
        
        # Load the model
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

def predict_multistep_direct(model, initial_context, variables, prediction_horizon=6, context_length=60):
    """
    Direct multi-step prediction that predicts all horizons simultaneously.
    This is the primary prediction method for multi-step models.
    
    Args:
        model: Trained multi-step model
        initial_context: Input context of shape (context_length, num_features)
        variables: List of variable names
        prediction_horizon: Number of steps to predict
        context_length: Length of context window
    
    Returns:
        predictions: List of prediction arrays, one for each future timestep
    """
    context_reshaped = initial_context.reshape(1, context_length, len(variables))
    
    # Make single prediction for all horizons
    multistep_prediction = model.predict(context_reshaped, verbose=0)
    
    # Reshape output to separate each time step
    n_features = len(variables)
    
    if multistep_prediction.shape[1] == prediction_horizon * n_features:
        # Model was trained for multi-step output
        predictions = []
        for step in range(prediction_horizon):
            start_idx = step * n_features
            end_idx = (step + 1) * n_features
            step_pred = multistep_prediction[0, start_idx:end_idx]
            predictions.append(step_pred)
        return predictions
    else:
        raise ValueError(f"Model output shape {multistep_prediction.shape} doesn't match expected multi-step format")

def predict_multistep_regularized(model, initial_context, variables, prediction_horizon=6, context_length=60):
    """
    Regularized multi-step prediction with bounds checking to prevent unrealistic predictions.
    """
    # Get raw multi-step predictions
    predictions = predict_multistep_direct(model, initial_context, variables, prediction_horizon, context_length)
    
    # Calculate context statistics for regularization
    context_stats = {
        'mean': np.mean(initial_context, axis=0),
        'std': np.std(initial_context, axis=0),
        'min': np.min(initial_context, axis=0),
        'max': np.max(initial_context, axis=0)
    }
    
    # Apply regularization to each prediction step
    regularized_predictions = []
    
    for step, pred in enumerate(predictions):
        # Step factor: allow more deviation for later predictions
        step_factor = 1.0 + (step * 0.15)  # Gradual increase in tolerance
        
        # Define bounds based on historical context
        lower_bounds = context_stats['min'] - step_factor * context_stats['std']
        upper_bounds = context_stats['max'] + step_factor * context_stats['std']
        
        # Clip predictions to reasonable bounds
        pred_clipped = np.clip(pred, lower_bounds, upper_bounds)
        regularized_predictions.append(pred_clipped)
    
    return regularized_predictions

def predict_multistep_ensemble(model, initial_context, variables, prediction_horizon=6, context_length=60):
    """
    Ensemble approach combining direct multi-step prediction with trend analysis.
    """
    # Get direct multi-step predictions
    direct_predictions = predict_multistep_direct(model, initial_context, variables, prediction_horizon, context_length)
    
    # Get regularized predictions
    regularized_predictions = predict_multistep_regularized(model, initial_context, variables, prediction_horizon, context_length)
    
    # Simple trend extrapolation for ensemble
    trend_predictions = extrapolate_trend_multistep(initial_context, variables, prediction_horizon)
    
    # Combine predictions with adaptive weights
    ensemble_predictions = []
    
    for step in range(prediction_horizon):
        # Weights that favor model predictions but incorporate trend for stability
        weight_direct = max(0.1, 0.8 - step * 0.05)      # Decreases with horizon
        weight_regularized = min(0.7, 0.2 + step * 0.05)  # Increases with horizon  
        weight_trend = min(0.2, step * 0.02)              # Small but increasing
        
        # Normalize weights
        total_weight = weight_direct + weight_regularized + weight_trend
        weight_direct /= total_weight
        weight_regularized /= total_weight
        weight_trend /= total_weight
        
        # Combine predictions
        combined_pred = (weight_direct * direct_predictions[step] + 
                        weight_regularized * regularized_predictions[step] + 
                        weight_trend * trend_predictions[step])
        
        ensemble_predictions.append(combined_pred)
    
    return ensemble_predictions

def extrapolate_trend_multistep(context, variables, prediction_horizon=6):
    """
    Simple trend extrapolation for multi-step ensemble prediction.
    """
    predictions = []
    
    # Use last 10 timesteps to calculate trend
    recent_window = min(10, len(context))
    recent_data = context[-recent_window:]
    
    # Calculate linear trend for each feature
    trends = []
    for feature_idx in range(len(variables)):
        feature_data = recent_data[:, feature_idx]
        if len(feature_data) >= 2:
            # Simple linear regression
            x = np.arange(len(feature_data))
            trend = np.polyfit(x, feature_data, 1)[0]  # slope
        else:
            trend = 0.0
        trends.append(trend)
    
    trends = np.array(trends)
    last_values = context[-1, :]
    
    # Extrapolate trend for each step
    for step in range(1, prediction_horizon + 1):
        predicted_values = last_values + (trends * step)
        predictions.append(predicted_values)
    
    return predictions

def improve_multistep_model(initial_model, training_data, target_data, prediction_error_mae, 
                           error_history_buffer=None, prediction_horizon=6):
    """
    Improved model updating specifically designed for multi-step models.
    
    Args:
        initial_model: Multi-step model to improve
        training_data: Input context data
        target_data: Multi-step target data (flattened: horizon * features)
        prediction_error_mae: Current prediction error
        error_history_buffer: Buffer to track error history
        prediction_horizon: Number of prediction steps
    """
    debug_mode = False

    try:
        print(f"   Attempting multi-step model improvement (current MAE: {prediction_error_mae:.6f})")
        
        # Store original weights for potential rollback
        original_weights = initial_model.get_weights()
        
        # Convert target data to numpy for analysis
        if hasattr(target_data, 'numpy'):
            target_data_np = target_data.numpy()
        else:
            target_data_np = np.array(target_data)
        
        # Get current predictions for analysis
        current_prediction = initial_model.predict(training_data, verbose=0)
        if hasattr(current_prediction, 'numpy'):
            current_prediction_np = current_prediction.numpy()
        else:
            current_prediction_np = np.array(current_prediction)
        
        if debug_mode:
            print(f"Current target shape: {target_data_np.shape}")
            print(f"Current prediction shape: {current_prediction_np.shape}")
        
        # Calculate per-horizon errors (reshape to analyze each prediction step)
        n_features = target_data_np.shape[1] // prediction_horizon
        
        # Reshape for horizon analysis
        target_reshaped = target_data_np.reshape(-1, prediction_horizon, n_features)
        pred_reshaped = current_prediction_np.reshape(-1, prediction_horizon, n_features)
        
        # Calculate errors per horizon
        horizon_errors = []
        for h in range(prediction_horizon):
            horizon_mae = np.mean(np.abs(pred_reshaped[:, h, :] - target_reshaped[:, h, :]))
            horizon_errors.append(horizon_mae)
        
        if debug_mode:
            print(f"Horizon errors: {horizon_errors}")
        
        # Initialize error history buffer if not provided
        if error_history_buffer is None:
            error_history_buffer = {'errors': [], 'window_size': 10, 'horizon_errors': []}
        
        # Update error history
        overall_error = np.mean(horizon_errors)
        error_history_buffer['errors'].append(overall_error)
        error_history_buffer['horizon_errors'].append(horizon_errors.copy())
        
        if len(error_history_buffer['errors']) > error_history_buffer['window_size']:
            error_history_buffer['errors'].pop(0)
            error_history_buffer['horizon_errors'].pop(0)
        
        # Analyze error trends across horizons
        deteriorating_horizons = np.zeros(prediction_horizon, dtype=bool)
        if len(error_history_buffer['horizon_errors']) >= 6:
            recent_horizon_errors = np.mean(error_history_buffer['horizon_errors'][-3:], axis=0)
            older_horizon_errors = np.mean(error_history_buffer['horizon_errors'][-6:-3], axis=0)
            horizon_trend = recent_horizon_errors - older_horizon_errors
            deteriorating_horizons = horizon_trend > (np.std(horizon_errors) * 0.1)
        
        # Identify problematic horizons (later horizons typically have higher error)
        error_threshold = np.median(horizon_errors) + np.std(horizon_errors)
        problematic_horizons = (np.array(horizon_errors) > error_threshold) | deteriorating_horizons
        n_problematic = np.sum(problematic_horizons)
        
        if n_problematic > 0:
            print(f"   → {n_problematic}/{prediction_horizon} horizons need attention")
            print(f"   → Deteriorating horizons: {np.sum(deteriorating_horizons)}")
            print(f"   → Problematic horizons: {np.where(problematic_horizons)[0] + 1}")
        
        # Adaptive learning rate based on horizon performance
        base_lr = 0.0008  # Slightly lower for multi-step stability
        
        # Adjust learning rate based on problem severity
        if n_problematic > prediction_horizon * 0.5:  # If >50% horizons are problematic
            adaptive_lr = base_lr * 2.5
            epochs = 20
        elif n_problematic > 0:
            adaptive_lr = base_lr * 1.8
            epochs = 15
        else:
            adaptive_lr = base_lr
            epochs = 8
        
        # Create sample weights favoring recent data
        n_samples = len(training_data)
        decay_factor = 0.08
        time_weights = np.exp(-decay_factor * np.arange(n_samples)[::-1])
        sample_weights = time_weights / np.sum(time_weights)
        
        # Batch size adjustment
        min_batch_size = max(8, min(32, len(training_data) // 4))
        if n_problematic > prediction_horizon * 0.3:
            batch_size = min_batch_size  # Smaller batches for focused learning
        else:
            batch_size = min(64, len(training_data) // 2)
        
        print(f"   → Using adaptive LR: {adaptive_lr:.5f}, epochs: {epochs}, batch_size: {batch_size}")
        
        # Recreate the weighted loss function for recompilation
        def weighted_mse_loss(y_true, y_pred):
            """Weighted MSE loss that gives more importance to near-term predictions"""
            y_true_reshaped = keras.ops.reshape(y_true, (-1, prediction_horizon, n_features))
            y_pred_reshaped = keras.ops.reshape(y_pred, (-1, prediction_horizon, n_features))
            
            # Create weights that decrease with prediction distance
            horizon_weights = keras.ops.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
            horizon_weights = horizon_weights[:prediction_horizon]
            
            # Calculate weighted squared errors
            squared_errors = keras.ops.square(y_true_reshaped - y_pred_reshaped)
            weighted_errors = squared_errors * horizon_weights[None, :, None]
            
            return keras.ops.mean(weighted_errors)
        
        # Compile model with adaptive learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=adaptive_lr)
        initial_model.compile(
            optimizer=optimizer,
            loss=weighted_mse_loss,
            metrics=['mae']
        )
        
        # Train the model with sample weights
        try:
            time_start = time.time()

            history = initial_model.fit(
                training_data, 
                target_data, 
                epochs=epochs, 
                batch_size=batch_size,
                sample_weight=sample_weights,
                verbose=0
            )
            
            # Validate improvement
            new_prediction = initial_model.predict(training_data, verbose=0)
            if hasattr(new_prediction, 'numpy'):
                new_prediction_np = new_prediction.numpy()
            else:
                new_prediction_np = np.array(new_prediction)
            
            # Calculate new horizon errors
            new_pred_reshaped = new_prediction_np.reshape(-1, prediction_horizon, n_features)
            new_horizon_errors = []
            for h in range(prediction_horizon):
                horizon_mae = np.mean(np.abs(new_pred_reshaped[:, h, :] - target_reshaped[:, h, :]))
                new_horizon_errors.append(horizon_mae)
            
            new_overall_error = np.mean(new_horizon_errors)
            
            # Multi-criteria improvement assessment
            global_improvement = new_overall_error < prediction_error_mae
            worst_horizon_improved = np.max(new_horizon_errors) < np.max(horizon_errors)
            
            # Check if problematic horizons specifically improved
            problematic_improvement = False
            if n_problematic > 0:
                old_problematic_error = np.mean([horizon_errors[i] for i in range(prediction_horizon) if problematic_horizons[i]])
                new_problematic_error = np.mean([new_horizon_errors[i] for i in range(prediction_horizon) if problematic_horizons[i]])
                problematic_improvement = old_problematic_error > new_problematic_error
            
            # Stability check across horizons
            old_horizon_std = np.std(horizon_errors)
            new_horizon_std = np.std(new_horizon_errors)
            stability_check = new_horizon_std <= old_horizon_std * 1.2
            
            # Count improvement criteria met
            criteria_met = sum([
                global_improvement,
                worst_horizon_improved,
                problematic_improvement,
                stability_check
            ])
            
            # Decision logic
            required_criteria = 2 if n_problematic > 0 else 2
            
            time_stop = time.time()    
            print(f"   Multi-step model training and evaluation took {time_stop - time_start:.2f} seconds")  
            
            if criteria_met >= required_criteria:
                improvement_ratio = (prediction_error_mae - new_overall_error) / (prediction_error_mae + 1e-8)
                print(f"   ✓ Multi-step model improved: {prediction_error_mae:.6f} → {new_overall_error:.6f} (Δ{improvement_ratio*100:.1f}%)")
                print(f"   ✓ Criteria met: {criteria_met}/4 (global:{global_improvement}, worst:{worst_horizon_improved}, problematic:{problematic_improvement}, stable:{stability_check})")
                
                # Show horizon-specific improvements
                for h in range(prediction_horizon):
                    if horizon_errors[h] > new_horizon_errors[h]:
                        print(f"     t+{h+1} improved: {horizon_errors[h]:.6f} → {new_horizon_errors[h]:.6f}")
                
                # Update error history with successful improvement
                error_history_buffer['errors'][-1] = new_overall_error
                error_history_buffer['horizon_errors'][-1] = new_horizon_errors
                
                return initial_model
            else:
                # Revert to original weights if insufficient improvement
                initial_model.set_weights(original_weights)
                print(f"   ✗ Insufficient improvement: criteria {criteria_met}/{required_criteria}, reverting weights")
                print(f"     Global: {global_improvement}, Worst: {worst_horizon_improved}, Problematic: {problematic_improvement}, Stable: {stability_check}")
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

def rolling_buffer_multistep_learning_prediction_with_dash(initial_model, 
                                                          df_online, 
                                                          scalers, 
                                                          context_length,
                                                          df_removed_nans_forecasting, 
                                                          df_removed_nans_classification,
                                                          dash_plotter, 
                                                          variables, 
                                                          prediction_horizon=6, 
                                                          classification_model_path=None,
                                                          prediction_method='ensemble'):
    """
    Enhanced rolling buffer prediction with multi-step model and online learning.
    
    Parameters:
    -----------
    prediction_method : str, default='ensemble'
        Prediction strategy to use:
        - 'direct': Direct multi-step prediction (recommended for multi-step models)
        - 'regularized': Regularized multi-step with bounds checking
        - 'ensemble': Adaptive ensemble of multi-step methods
    """
    # Prepare model for online learning
    print("\n✓ Preparing multi-step model for online learning...")
    try:
        # Recreate the weighted loss function for recompilation
        def weighted_mse_loss(y_true, y_pred):
            n_features = len(variables)
            y_true_reshaped = keras.ops.reshape(y_true, (-1, prediction_horizon, n_features))
            y_pred_reshaped = keras.ops.reshape(y_pred, (-1, prediction_horizon, n_features))
            
            horizon_weights = keras.ops.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
            horizon_weights = horizon_weights[:prediction_horizon]
            
            squared_errors = keras.ops.square(y_true_reshaped - y_pred_reshaped)
            weighted_errors = squared_errors * horizon_weights[None, :, None]
            
            return keras.ops.mean(weighted_errors)
        
        initial_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001), 
            loss=weighted_mse_loss, 
            metrics=['mae']
        )
        print("✓ Multi-step model successfully prepared for online learning")
    except Exception as e:
        print(f"Warning: Could not recompile multi-step model ({e}), will try per-step recompilation")
    
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
        
        # Set the label_to_name dictionary in the dashboard
        if dash_plotter is not None:
            dash_plotter.set_label_to_name_dict(label_to_name)

    except Exception as e:
        print(f"Warning: Could not load classification model: {e}")
        print("   Continuing with multi-step forecasting only...")
        classification_enabled = False
        classification_model = None

    # Error history buffer for model improvement
    error_history_buffer = None

    # Main prediction loop
    for t in range(context_length, len(scaled_data) - prediction_horizon + 1):
        # Wait 1 second (for demo - real data is 1 minute apart)
        time.sleep(1)
        current_step = t - context_length

        # 1. Make multi-step predictions using selected method
        try:
            if prediction_method == 'ensemble':
                step_predictions = predict_multistep_ensemble(model, current_context, variables=variables, 
                                                            prediction_horizon=prediction_horizon, context_length=context_length)
                print(f"   Using multi-step ensemble prediction for step {current_step}")
            elif prediction_method == 'regularized':
                step_predictions = predict_multistep_regularized(model, current_context, variables=variables, 
                                                               prediction_horizon=prediction_horizon, context_length=context_length)
                print(f"   Using regularized multi-step prediction for step {current_step}")
            else:  # Default to direct
                step_predictions = predict_multistep_direct(model, current_context, variables=variables, 
                                                          prediction_horizon=prediction_horizon, context_length=context_length)
                print(f"   Using direct multi-step prediction for step {current_step}")
                
        except Exception as e:
            print(f"   {prediction_method} multi-step prediction failed ({e}), falling back to direct")
            # Fallback to direct prediction if selected method fails
            step_predictions = predict_multistep_direct(model, current_context, variables=variables, 
                                                      prediction_horizon=prediction_horizon, context_length=context_length)
        
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

        # Port status check
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
                
                # Pass the ACTUAL prediction for t+1
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
                    
                except Exception as e:
                    print(f"DEBUG: Error in dash plotter: {e}")

        # 8. Update context and model for next iteration
        new_row = scaled_data[t, :].copy()
        
        # Prepare training data for multi-step model improvement
        if t + prediction_horizon <= len(scaled_data):  # Ensure we have ground truth for all horizons
            # Context: up to current step t
            training_context = np.vstack((current_context[1:], new_row))
            
            # Multi-step target: next prediction_horizon steps (flattened)
            multistep_target = []
            for step in range(prediction_horizon):
                if t + 1 + step < len(scaled_data):
                    multistep_target.extend(scaled_data[t + 1 + step, :])
            
            if len(multistep_target) == prediction_horizon * len(variables):
                # Calculate current prediction error for the improve_model function
                current_pred = model.predict(
                    training_context.reshape(1, context_length, len(variables)), 
                    verbose=0
                )
                
                # Calculate MAE across all prediction horizons
                target_array = np.array(multistep_target).reshape(1, -1)
                current_prediction_error = np.mean(np.abs(current_pred - target_array))
                
                # Enhanced multi-step model improvement
                model = improve_multistep_model(
                    model, 
                    training_context.reshape(1, context_length, len(variables)), 
                    target_array, 
                    prediction_error_mae=current_prediction_error,
                    error_history_buffer=error_history_buffer,
                    prediction_horizon=prediction_horizon
                )

                print(f"Multi-step model improved at step {current_step}: training to predict t+1 to t+{prediction_horizon}")
        
        # Update context for next iteration
        current_context = np.vstack((current_context[1:], new_row))
        
        # 9. Exit on 'Q' key press
        if keyboard.is_pressed('q'):
            print("Exiting multi-step program...")
            break

    print("=" * 60)
    print("Multi-step rolling prediction completed!")

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