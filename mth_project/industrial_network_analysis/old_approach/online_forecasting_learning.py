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
        
        # FIXED: Create new row directly from prediction
        # This is the predicted next timestep
        new_row = step_prediction[0].copy()  # Extract from batch dimension and copy
        
        # Slide context window: remove oldest, add new prediction
        context = np.vstack((context[1:], new_row.reshape(1, -1)))
    
    return predictions

def predict_direct_multistep(model, initial_context, variables, prediction_horizon=6, context_length=60):
    """
    Direct multi-step prediction that predicts all horizons simultaneously.
    Avoids error accumulation from recursive prediction.
    
    Note: This requires a model trained to output prediction_horizon * n_features values.
    """
    context_reshaped = initial_context.reshape(1, context_length, len(variables))
    
    # Make single prediction for all horizons
    multistep_prediction = model.predict(context_reshaped, verbose=0)
    
    # Reshape output to separate each time step
    # Expected shape: (1, prediction_horizon * n_features)
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
        # Fallback to recursive if model doesn't support multi-step
        print("   Warning: Model not configured for multi-step, using recursive fallback")
        return predict_recursive_steps(model, initial_context, variables, prediction_horizon, context_length)

def predict_regularized_recursive(model, initial_context, variables, prediction_horizon=6, context_length=60):
    """
    Regularized recursive prediction with bounds checking to prevent drift.
    """
    context = initial_context.copy()
    predictions = []
    
    # Calculate context statistics for regularization
    context_stats = {
        'mean': np.mean(context, axis=0),
        'std': np.std(context, axis=0),
        'min': np.min(context, axis=0),
        'max': np.max(context, axis=0)
    }
    
    for step in range(prediction_horizon):
        context_reshaped = context.reshape(1, context_length, len(variables))
        step_prediction = model.predict(context_reshaped, verbose=0)
        
        # Regularization: clip predictions to reasonable bounds
        pred_flat = step_prediction.flatten()
        
        # Define bounds based on historical context (more lenient for later steps)
        step_factor = 1.0 + (step * 0.2)  # Allow more deviation for later predictions
        lower_bounds = context_stats['min'] - step_factor * context_stats['std']
        upper_bounds = context_stats['max'] + step_factor * context_stats['std']
        
        # Clip predictions
        pred_clipped = np.clip(pred_flat, lower_bounds, upper_bounds)
        
        predictions.append(pred_clipped)
        
        # Update context with clipped prediction
        new_row = pred_clipped.reshape(1, -1)
        context = np.vstack((context[1:], new_row))
    
    return predictions

def extrapolate_trend(context, variables, prediction_horizon=6):
    """
    Simple trend extrapolation for ensemble prediction.
    Uses linear trend from recent context window.
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
    
    # Extrapolate trend
    for step in range(1, prediction_horizon + 1):
        predicted_values = last_values + (trends * step)
        predictions.append(predicted_values)
    
    return predictions

def predict_ensemble(model, initial_context, variables, prediction_horizon=6, context_length=60):
    """
    Ensemble prediction combining multiple strategies with adaptive weights.
    """
    # Get predictions from different methods
    pred_recursive = predict_recursive_steps(model, initial_context, variables, prediction_horizon, context_length)
    pred_regularized = predict_regularized_recursive(model, initial_context, variables, prediction_horizon, context_length)
    pred_trend = extrapolate_trend(initial_context, variables, prediction_horizon)
    
    # Adaptive weights based on prediction horizon
    ensemble_predictions = []
    
    for step in range(prediction_horizon):
        # Weights that change based on prediction step
        weight_recursive = max(0.1, 1.0 - step * 0.15)     # Strong for early steps
        weight_regularized = min(0.8, 0.3 + step * 0.1)    # Stronger for later steps
        weight_trend = min(0.3, step * 0.05)               # Increases with step
        
        # Normalize weights
        total_weight = weight_recursive + weight_regularized + weight_trend
        weight_recursive /= total_weight
        weight_regularized /= total_weight
        weight_trend /= total_weight
        
        # Combine predictions
        combined_pred = (weight_recursive * pred_recursive[step] + 
                        weight_regularized * pred_regularized[step] + 
                        weight_trend * pred_trend[step])
        
        ensemble_predictions.append(combined_pred)
    
    return ensemble_predictions

def improve_model(initial_model, training_data, target_data, prediction_error_mae, error_history_buffer=None, multistep_targets=None):

    debug_mode = False

    try:

        print(f"   Attempting model improvement (current MAE: {prediction_error_mae:.6f})")
        
        # Store original weights for potential rollback
        original_weights = initial_model.get_weights()
        
        # Convert target data to numpy for analysis
        if hasattr(target_data, 'numpy'):
            target_data_np = target_data.numpy()
        else:
            target_data_np = np.array(target_data)
            
        # Get current predictions for per-feature analysis
        current_prediction = initial_model.predict(training_data, verbose=0)
        if hasattr(current_prediction, 'numpy'):
            current_prediction_np = current_prediction.numpy()
        else:
            current_prediction_np = np.array(current_prediction)
        
        if debug_mode:
            print(f"Current target sample shape: {target_data_np.shape}")
            print(f"Current prediction sample shape: {current_prediction_np.shape}")
            
            print("   Current target sample:", target_data_np[0])
            print("   Current prediction sample:", current_prediction_np[0])

        # Calculate per-feature errors
        per_feature_errors = np.mean(np.abs(current_prediction_np - target_data_np), axis=0)
        n_features = len(per_feature_errors)
        
        if debug_mode:
            print(f" current per_feature_error shape: {per_feature_errors.shape}")
            print("   Current per-feature errors:", per_feature_errors)

        # Initialize error history buffer if not provided
        if error_history_buffer is None:
            error_history_buffer = {'errors': [], 'window_size': 10}
        
        # Update error history
        error_history_buffer['errors'].append(per_feature_errors.copy())
        if len(error_history_buffer['errors']) > error_history_buffer['window_size']:
            error_history_buffer['errors'].pop(0)
        
        # Analyze error trends (if we have enough history)
        deteriorating_features = np.zeros(n_features, dtype=bool)
        if len(error_history_buffer['errors']) >= 6:
            recent_errors = np.mean(error_history_buffer['errors'][-3:], axis=0)
            older_errors = np.mean(error_history_buffer['errors'][-6:-3], axis=0)
            error_trend = recent_errors - older_errors
            deteriorating_features = error_trend > (np.std(per_feature_errors) * 0.1)
        
        # Identify problematic features (high error or deteriorating)
        error_threshold = np.median(per_feature_errors) + np.std(per_feature_errors)
        problematic_features = (per_feature_errors > error_threshold) | deteriorating_features
        n_problematic = np.sum(problematic_features)
        
        if n_problematic > 0:
            print(f"   → {n_problematic}/{n_features} features need attention")
            print(f"   → Deteriorating features: {np.sum(deteriorating_features)}")
        
        # Adaptive learning rate based on feature performance
        base_lr = 0.001
        feature_severity = per_feature_errors / (np.mean(per_feature_errors) + 1e-8)
        
        # Higher learning rate for worse-performing features
        if n_problematic > n_features * 0.3:  # If >30% features are problematic
            adaptive_lr = base_lr * 2.0
            epochs = 15
        elif n_problematic > 0:
            adaptive_lr = base_lr * 1.5
            epochs = 10
        else:
            adaptive_lr = base_lr
            epochs = 5
        
        # Create recency weights (more recent samples get higher weight)
        n_samples = len(training_data)
        decay_factor = 0.1
        time_weights = np.exp(-decay_factor * np.arange(n_samples)[::-1])
        sample_weights = time_weights / np.sum(time_weights)
        
        # Determine batch size based on data size and problematic features
        min_batch_size = max(8, min(32, len(training_data) // 4))
        if n_problematic > n_features * 0.5:
            batch_size = min_batch_size  # Smaller batches for focused learning
        else:
            batch_size = min(64, len(training_data) // 2)
        
        print(f"   → Using adaptive LR: {adaptive_lr:.5f}, epochs: {epochs}, batch_size: {batch_size}")
        
        # Compile model with adaptive learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=adaptive_lr)
        initial_model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['accuracy', 'mae']
        )
        
        # Multi-step training if multistep targets are provided
        if multistep_targets is not None and len(multistep_targets) > 1:
            print(f"   → Training with multi-step targets ({len(multistep_targets)} steps)")
            
            # Create multi-step target by concatenating all future steps
            multistep_target_combined = np.concatenate(multistep_targets, axis=1)
            
            # Train with both single-step and multi-step objectives
            # Use smaller learning rate for multi-step training
            multistep_lr = adaptive_lr * 0.5
            multistep_optimizer = tf.keras.optimizers.Adam(learning_rate=multistep_lr)
            
            # Clone model for multi-step training
            multistep_model = tf.keras.models.clone_model(initial_model)
            multistep_model.set_weights(initial_model.get_weights())
            
            # Modify last layer for multi-step output if necessary
            if multistep_model.output_shape[1] != multistep_target_combined.shape[1]:
                print(f"   → Adapting model output for multi-step: {multistep_model.output_shape[1]} → {multistep_target_combined.shape[1]}")
                # For now, continue with single-step training
                # TODO: Implement dynamic model adaptation
            else:
                multistep_model.compile(
                    optimizer=multistep_optimizer,
                    loss='mse',
                    metrics=['mae']
                )
                
                try:
                    # Train multi-step model
                    multistep_history = multistep_model.fit(
                        training_data,
                        multistep_target_combined,
                        epochs=max(1, epochs // 2),
                        batch_size=batch_size,
                        sample_weight=sample_weights,
                        verbose=0
                    )
                    
                    # Test multi-step model performance
                    multistep_pred = multistep_model.predict(training_data, verbose=0)
                    multistep_error = np.mean(np.abs(multistep_pred - multistep_target_combined))
                    
                    # If multi-step training is better, use it
                    if multistep_error < prediction_error_mae * 1.1:  # Allow 10% tolerance
                        print(f"   ✓ Multi-step training successful: {multistep_error:.6f}")
                        initial_model.set_weights(multistep_model.get_weights())
                    else:
                        print(f"   ✗ Multi-step training not beneficial: {multistep_error:.6f}")
                        
                except Exception as e:
                    print(f"   Warning: Multi-step training failed: {e}")
        
        # Train the model with sample weights for recency bias
        try:
            time_start = time.time()

            history = initial_model.fit(
                training_data, 
                target_data, 
                epochs=epochs, 
                batch_size=batch_size,
                sample_weight=sample_weights,  # Give more weight to recent samples
                verbose=0
            )
            
            # Validate improvement with multi-criteria assessment
            new_prediction = initial_model.predict(training_data, verbose=0)
            if hasattr(new_prediction, 'numpy'):
                new_prediction_np = new_prediction.numpy()
            else:
                new_prediction_np = np.array(new_prediction)
            
            # Calculate new per-feature errors
            new_per_feature_errors = np.mean(np.abs(new_prediction_np - target_data_np), axis=0)
            new_global_error = np.mean(new_per_feature_errors)
            
            # Multi-criteria improvement assessment
            global_improvement = new_global_error < prediction_error_mae
            worst_feature_improved = np.max(new_per_feature_errors) < np.max(per_feature_errors)
            
            # Check if problematic features specifically improved
            problematic_improvement = 0
            if n_problematic > 0:
                old_problematic_error = np.mean(per_feature_errors[problematic_features])
                new_problematic_error = np.mean(new_per_feature_errors[problematic_features])
                problematic_improvement = old_problematic_error > new_problematic_error
            
            # Stability check (new predictions shouldn't be too volatile)
            stability_check = np.std(new_per_feature_errors) <= np.std(per_feature_errors) * 1.2
            
            # Count improvement criteria met
            criteria_met = sum([
                global_improvement,
                worst_feature_improved,
                problematic_improvement,
                stability_check
            ])
            
            # Decision logic: accept if enough criteria are met
            required_criteria = 2 if n_problematic > 0 else 2
            
            time_stop = time.time()    
            print(f"   Model training and evaluation took {time_stop - time_start:.2f} seconds")  
            
            if criteria_met >= required_criteria:
                improvement_ratio = (prediction_error_mae - new_global_error) / (prediction_error_mae + 1e-8)
                print(f"   ✓ Model improved: {prediction_error_mae:.6f} → {new_global_error:.6f} (Δ{improvement_ratio*100:.1f}%)")
                print(f"   ✓ Criteria met: {criteria_met}/4 (global:{global_improvement}, worst:{worst_feature_improved}, problematic:{problematic_improvement}, stable:{stability_check})")
                
                # Update error history with successful improvement
                error_history_buffer['errors'][-1] = new_per_feature_errors
                
                return initial_model
            else:
                # Revert to original weights if insufficient improvement
                initial_model.set_weights(original_weights)
                print(f"   ✗ Insufficient improvement: criteria {criteria_met}/{required_criteria}, reverting weights")
                print(f"     Global: {global_improvement}, Worst: {worst_feature_improved}, Problematic: {problematic_improvement}, Stable: {stability_check}")
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
                                        classification_model_path=None,
                                        prediction_method='ensemble'):
    """
    Enhanced rolling buffer prediction with multiple forecasting strategies.
    
    Parameters:
    -----------
    prediction_method : str, default='ensemble'
        Prediction strategy to use:
        - 'recursive': Original recursive prediction
        - 'regularized': Regularized recursive with bounds checking
        - 'ensemble': Adaptive ensemble of multiple methods (recommended)
        - 'direct': Direct multi-step prediction (requires compatible model)
    """
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
        print("   Continuing with forecasting only...")
        classification_enabled = False
        classification_model = None

    # for improved model
    feature_states = None

    # Main prediction loop
    for t in range(context_length, len(scaled_data) - prediction_horizon + 1):
        # wait 1 seconds (for demo - real data is 1 minute apart)
        time.sleep(1)
        current_step = t - context_length

        # 1. Make predictions for next 'prediction_horizon' steps using selected method
        try:
            if prediction_method == 'ensemble':
                step_predictions = predict_ensemble(model, current_context, variables=variables, prediction_horizon=prediction_horizon)
                print(f"   Using ensemble prediction for step {current_step}")
            elif prediction_method == 'regularized':
                step_predictions = predict_regularized_recursive(model, current_context, variables=variables, prediction_horizon=prediction_horizon)
                print(f"   Using regularized recursive prediction for step {current_step}")
            elif prediction_method == 'direct':
                step_predictions = predict_direct_multistep(model, current_context, variables=variables, prediction_horizon=prediction_horizon)
                print(f"   Using direct multi-step prediction for step {current_step}")
            else:  # Default to recursive
                step_predictions = predict_recursive_steps(model, current_context, variables=variables, prediction_horizon=prediction_horizon)
                print(f"   Using recursive prediction for step {current_step}")
                
        except Exception as e:
            print(f"   {prediction_method} prediction failed ({e}), falling back to recursive")
            # Fallback to recursive prediction if selected method fails
            step_predictions = predict_recursive_steps(model, current_context, variables=variables, prediction_horizon=prediction_horizon)
        
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
            
            # Prepare multi-step targets if we have enough future data
            multistep_targets = []
            max_steps = min(prediction_horizon, len(scaled_data) - t - 1)
            
            for step_ahead in range(1, max_steps + 1):
                if t + step_ahead < len(scaled_data):
                    future_target = scaled_data[t + step_ahead, :].copy()
                    multistep_targets.append(future_target.reshape(1, len(variables)))
            
            # Train model to predict t+1 from context ending at t
            if t + 1 < len(scaled_data):
                # Calculate current prediction error for the improve_model function
                current_pred = model.predict(
                    training_context.reshape(1, context_length, len(variables)), 
                    verbose=0
                )
                current_prediction_error = np.mean(np.abs(current_pred.flatten() - target_next_step))
                
                # Enhanced model improvement with multi-step training
                model = improve_model(
                    model, 
                    training_context.reshape(1, context_length, len(variables)), 
                    target_next_step.reshape(1, len(variables)), 
                    prediction_error_mae=current_prediction_error,
                    multistep_targets=multistep_targets if len(multistep_targets) > 1 else None
                )

            print(f"Model improved at step {current_step}: training to predict t+1 (with {len(multistep_targets)} multi-step targets)")
        
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