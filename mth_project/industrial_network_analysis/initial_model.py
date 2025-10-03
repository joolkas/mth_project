from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import time
import sys
import os
import pickle
from get_data import get_processed_path

import matplotlib.pyplot as plt

# Import callbacks at module level to avoid conflicts
try:
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
except ImportError:
    # Fallback for different TensorFlow versions
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# model mode

model_mode = "multi_step"  # "one_step" or "multi_step"

# model parameters
epochs = 80
batch_size = 64
validation_split = 0.2
verbose = 1
context_length = 60    # Input window length
prediction_horizon = 6  # Number of future steps to predict directly
first_layer_units = 128
second_layer_units = 64
third_layer_units = 32  # Additional layer for multi-step complexity
dense_units = 256       # Larger dense layer for multi-step output
activation = 'relu'
dropout_rate = 0.3

# for model training
use_callbacks = True
early_stopping_patience = 10  # More patience for complex multi-step model
reduce_lr_factor = 0.5       # Less aggressive learning rate reduction
reduce_lr_patience = 5       # More patience for learning rate reduction

# Split data for initial training and online forecasting
initial_idx = 48 * 60 # first 48 hours for initial training
split_ratio = 0.8  # 80% training, 20% testing

# Use relative path from current file location
initial_model_path = os.path.join(os.path.dirname(__file__), "forecasting_model")  # path for model saving and loading

model_description = f"Initial Model - Epochs: {epochs},\n \
    Batch Size: {batch_size},\n Validation Split: {validation_split},\n Context Length: {context_length},\n \
    Prediction Horizon: {prediction_horizon},\n First Layer Units: {first_layer_units},\n Second Layer Units: {second_layer_units},\n \
    Third Layer Units: {third_layer_units},\n Dense Units: {dense_units},\n Activation: {activation},\n Dropout Rate: {dropout_rate}\n \
    Initial Training Samples: {initial_idx},"

results_file_name = "initial_model_results_006"

def create_online_multistep_model_simple(context_length, num_features, prediction_horizon):
    """Simplified multi-step model creation for compatibility recovery"""
    output_size = prediction_horizon * num_features
    
    model = keras.models.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=(context_length, num_features)),
        keras.layers.Dropout(0.15),  
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.15),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dropout(0.21),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.15),
        keras.layers.Dense(output_size, activation='linear')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001), 
        loss='mse',
        metrics=['accuracy','mse']
    )
    
    return model

def create_online_onestep_model_simple(context_length, num_features):
    """Simplified one-step model creation for compatibility recovery"""
    model = keras.models.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=(context_length, num_features)),
        keras.layers.Dropout(0.15),  
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.21),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.15),
        keras.layers.Dense(num_features, activation='linear')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001), 
        loss='mse',
        metrics=['accuracy', 'mse']
    )
    
    return model

def create_online_multistep_model(df,
                                  context_length=context_length,
                                  prediction_horizon=prediction_horizon,
                                  first_layer_units=first_layer_units,
                                  second_layer_units=second_layer_units,
                                  third_layer_units=third_layer_units,
                                  dense_units=dense_units,
                                  activation=activation,
                                  dropout_rate=dropout_rate):
    
    num_features = len(df.columns)
    
    # Multi-step output: prediction_horizon * num_features
    output_size = prediction_horizon * num_features
    
    model = keras.models.Sequential([
        keras.layers.LSTM(first_layer_units, return_sequences=True, input_shape=(context_length, num_features)),
        keras.layers.Dropout(dropout_rate * 0.5),  

        keras.layers.LSTM(second_layer_units, return_sequences=True),
        keras.layers.Dropout(dropout_rate * 0.5),

        keras.layers.LSTM(third_layer_units, return_sequences=False),
        keras.layers.Dropout(dropout_rate * 0.7),

        keras.layers.Dense(dense_units, activation=activation),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Dense(dense_units // 2, activation=activation),
        keras.layers.Dropout(dropout_rate * 0.5),

        keras.layers.Dense(output_size, activation='linear')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001), 
        loss='mse',
        metrics=['accuracy','mse']
    )
    
    return model

def create_online_onestep_model(df,
                                  context_length=context_length,
                                  first_layer_units=first_layer_units,
                                  second_layer_units=second_layer_units,
                                  dense_units=dense_units,
                                  activation=activation,
                                  dropout_rate=dropout_rate):
    
    num_features = len(df.columns)
    
    model = keras.models.Sequential([
        keras.layers.LSTM(first_layer_units, return_sequences=True, input_shape=(context_length, num_features)),
        keras.layers.Dropout(dropout_rate * 0.5),  

        keras.layers.LSTM(second_layer_units, return_sequences=False),
        keras.layers.Dropout(dropout_rate * 0.7),

        keras.layers.Dense(dense_units, activation=activation),
        keras.layers.Dropout(dropout_rate),

        keras.layers.Dense(dense_units // 2, activation=activation),
        keras.layers.Dropout(dropout_rate * 0.5),

        keras.layers.Dense(num_features, activation='linear')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001), 
        loss='mse',
        metrics=['accuracy', 'mse']
    )
    
    return model

def split_data_for_multistep_model(df, context_length=context_length, prediction_horizon=prediction_horizon):
    # store scalers for each variable
    scalers = {}
    scaled_data = np.zeros_like(df.values)

    # scale each column separately
    for i, var in enumerate(df.columns):
        scaler = StandardScaler()
        scaled_data[:, i] = scaler.fit_transform(df[var].values.reshape(-1, 1)).flatten()
        # Store the scaler for inverse transform
        scalers[var] = scaler  

    # convert back to DataFrame with original column names and index
    df_scaled = pd.DataFrame(
        data=scaled_data,
        columns=df.columns,
        index=df.index
    )

    # create training sequences for multi-step prediction
    X_train, y_train = [], []
    original_indices = []

    # we need context_length + prediction_horizon data points to create one sample
    for i in range(context_length, len(df_scaled) - prediction_horizon + 1):
        # Input: context_length timesteps
        X_train.append(df_scaled.iloc[i-context_length:i].values)
        
        # Output: next prediction_horizon timesteps (flattened)
        future_steps = []
        for step in range(prediction_horizon):
            future_steps.extend(df_scaled.iloc[i + step].values)
        y_train.append(future_steps)
        original_indices.append(i)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Multi-step training data shapes:")
    print(f"  X_train: {X_train.shape} (samples, context_length, features)")
    print(f"  y_train: {y_train.shape} (samples, prediction_horizon * features)")

    return X_train, y_train, scalers, original_indices

def split_data_for_onestep_model(df, context_length=context_length):
    # store scalers for each variable
    scalers = {}
    scaled_data = np.zeros_like(df.values)

    # scale each column separately
    for i, var in enumerate(df.columns):
        scaler = StandardScaler()
        scaled_data[:, i] = scaler.fit_transform(df[var].values.reshape(-1, 1)).flatten()
        # Store the scaler for inverse transform
        scalers[var] = scaler  

    # convert back to DataFrame with original column names and index
    df_scaled = pd.DataFrame(
        data=scaled_data,
        columns=df.columns,
        index=df.index
    )

    # create training sequences for one-step prediction
    X_train, y_train = [], []
    original_indices = []

    # we need context_length + 1 data points to create one sample
    for i in range(context_length, len(df_scaled)):
        # Input: context_length timesteps
        X_train.append(df_scaled.iloc[i-context_length:i].values)
        
        # Output: next single timestep
        y_train.append(df_scaled.iloc[i].values)
        original_indices.append(i)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"One-step training data shapes:")
    print(f"  X_train: {X_train.shape} (samples, context_length, features)")
    print(f"  y_train: {y_train.shape} (samples, features)")

    return X_train, y_train, scalers, original_indices

def train_model(model, 
                X_train,
                y_train, 
                epochs = epochs,
                batch_size = batch_size,
                validation_split = validation_split,
                verbose = verbose,
                use_callbacks = use_callbacks,
                es_patience = early_stopping_patience,
                lr_factor = reduce_lr_factor,
                lr_patience = reduce_lr_patience,
                model_save_path = os.path.join(os.path.dirname(__file__), "forecasting_model", "best_model.h5")):
    
    callback_list = []
    if use_callbacks:
        callback_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=es_patience,  # More patience for complex multi-step model
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=lr_factor,  # Less aggressive learning rate reduction
                patience=lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path if model_save_path else "best_model.h5",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
        callbacks=callback_list if use_callbacks else None
    )

    return history, model

def inverse_difference(predictions_arrays, last_actual_values):
    """Convert differenced predictions back to actual values."""
    actual_predictions = []
    current_values = last_actual_values.copy()
    
    for pred_diff in predictions_arrays:
        # Add difference to get actual value
        current_values = current_values + np.array(pred_diff)
        actual_predictions.append(current_values.copy())
    
    return actual_predictions

def predict_multistep_direct(model, context, variables, prediction_horizon=prediction_horizon):
    # Reshape context for model input
    context_reshaped = context.reshape(1, context.shape[0], len(variables))
    
    # Get multi-step prediction
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

def test_model(model, df, X_test, y_test, scalers, df_removed_nans_forecasting, 
                        test_indices, test_mode = "multi_step", prediction_horizon=prediction_horizon):
    predictions_test = []
    actuals_test = []
    
    n_features = len(df.columns)

    for i in range(len(y_test)):
        # Get context for prediction
        context = X_test[i]
        
        if test_mode == "multi_step":
            # Make multi-step prediction
            step_predictions = predict_multistep_direct(model, context, df.columns, prediction_horizon)
            
            # Convert y_test back to multi-step format for comparison
            actual_steps = []
            for step in range(prediction_horizon):
                start_idx = step * n_features
                end_idx = (step + 1) * n_features
                actual_steps.append(y_test[i][start_idx:end_idx])
                
        else:  # one_step mode
            # Make one-step prediction
            context_reshaped = context.reshape(1, context.shape[0], n_features)
            pred = model.predict(context_reshaped, verbose=0)
            step_predictions = [pred[0]]  # Wrap in list for consistency
            
            # y_test is already in correct format for one-step (no reshaping needed)
            actual_steps = [y_test[i]]  # Wrap in list for consistency
        
        predictions_test.append(step_predictions)
        actuals_test.append(actual_steps)

    # Convert to arrays and inverse transform
    all_predictions = []
    all_actuals = []
    
    for sample_idx in range(len(predictions_test)):
        sample_preds = []
        sample_actuals = []
        
        # Determine how many steps to process based on mode
        steps_to_process = prediction_horizon if test_mode == "multi_step" else 1
        
        for step in range(steps_to_process):
            # Inverse transform predictions
            pred_original = np.zeros(n_features)
            actual_original = np.zeros(n_features)
            
            for feat_idx, var in enumerate(df.columns):
                scaler = scalers[var]
                pred_original[feat_idx] = scaler.inverse_transform(
                    [[predictions_test[sample_idx][step][feat_idx]]])[0, 0]
                actual_original[feat_idx] = scaler.inverse_transform(
                    [[actuals_test[sample_idx][step][feat_idx]]])[0, 0]
            
            sample_preds.append(pred_original)
            sample_actuals.append(actual_original)
        
        all_predictions.append(sample_preds)
        all_actuals.append(sample_actuals)

    # Use only the first step for evaluation (t+1 predictions)
    first_step_predictions = [pred[0] for pred in all_predictions]
    first_step_actuals = [actual[0] for actual in all_actuals]

    # Apply inverse differencing
    predictions_original = []
    actuals_original = []
    
    for sample_idx in range(len(first_step_predictions)):
        # Get the original index for this test sample
        original_idx = test_indices[sample_idx]
        
        # Get the baseline value from the original (non-differenced) data
        # We need the value just before the prediction starts
        baseline_idx = original_idx - 1  # One step back from prediction start
        baseline_values = df_removed_nans_forecasting.iloc[baseline_idx][df.columns].values
        
        # Apply inverse differencing for this specific sample
        pred_original = inverse_difference([first_step_predictions[sample_idx]], baseline_values)
        actual_original = inverse_difference([first_step_actuals[sample_idx]], baseline_values)
        
        predictions_original.extend(pred_original)
        actuals_original.extend(actual_original)

    # Convert to DataFrames
    actuals_df = pd.DataFrame(data=actuals_original, columns=df.columns)
    predictions_df = pd.DataFrame(data=predictions_original, columns=df.columns)

    return actuals_df, predictions_df, all_actuals, all_predictions

def calculate_metrics(df, actuals_original, predictions_original, all_actuals, all_predictions, 
                               prediction_horizon=prediction_horizon, mode="multi_step"):
    """
    Calculate metrics for multi-step or one-step predictions including horizon-specific performance.
    """
    # Overall metrics (using first step)
    mse = mean_squared_error(actuals_original, predictions_original)
    mae = mean_absolute_error(actuals_original, predictions_original)
    rmse = np.sqrt(mse)
    percentage_error = np.mean(np.abs((actuals_original - predictions_original) / actuals_original)) * 100

    model_type = "Multi-Step" if mode == "multi_step" else "One-Step"
    print(f"\n{model_type} Model Performance (t+1 predictions):")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Percentage Error: {percentage_error:.6f}")

    # Calculate metrics for each prediction horizon (only for multi-step)
    horizon_metrics = []
    if mode == "multi_step":
        for horizon in range(prediction_horizon):
            horizon_predictions = [pred[horizon] for pred in all_predictions]
            horizon_actuals = [actual[horizon] for actual in all_actuals]
            
            horizon_mse = mean_squared_error(horizon_actuals, horizon_predictions)
            horizon_mae = mean_absolute_error(horizon_actuals, horizon_predictions)
            
            horizon_metrics.append({
                'horizon': horizon + 1,
                'mse': horizon_mse,
                'mae': horizon_mae,
                'rmse': np.sqrt(horizon_mse)
            })
            
            print(f"  t+{horizon+1} - MSE: {horizon_mse:.6f}, MAE: {horizon_mae:.6f}")

    # Show sample results
    results_df = pd.DataFrame({
        'Variable': df.columns.tolist() * actuals_original.shape[0],
        'Sample': [i for i in range(actuals_original.shape[0]) for _ in df.columns],
        'Actual': actuals_original.values.flatten(),
        'Predicted': predictions_original.values.flatten(),
        'Error': (actuals_original.values - predictions_original.values).flatten()
    })

    print(f"\nSample Results (t+1):")
    print(results_df.head(20))
    
    # Return horizon_metrics for multi-step, empty list for one-step
    return results_df, mse, mae, rmse, percentage_error, horizon_metrics

def plot_results(actuals_df, predictions_df, title, history=None):
    
    if history is None:
        print("No training history provided")
    else:
        # plot accuracy, loss and mse
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

        plt.title(f'Model {title} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model {title} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(history.history['mse'], label='Train MSE')
        plt.plot(history.history['val_mse'], label='Validation MSE')
        plt.title(f'Model {title} MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()

        plt.tight_layout()
        plt.show()


        # Create reasonable figure size: max 20 inches wide, 4 inches per subplot
        n_columns = len(actuals_df.columns)
        fig_width = min(20, max(12, n_columns * 2))  # Between 12-20 inches wide
        fig_height = n_columns * 4  # 4 inches per subplot
        
        plt.figure(figsize=(fig_width, fig_height))
        for i, column in enumerate(actuals_df.columns):
            plt.subplot(len(actuals_df.columns), 1, i+1)
            
            # Plot actual values
            plt.plot(actuals_df.index, actuals_df[column], 
                    label=f'Actual {column}', color='blue', linewidth=2, alpha=0.8)
            
            # Plot predicted values
            plt.plot(predictions_df.index, predictions_df[column], 
                    label=f'Predicted {column}', color='red', linewidth=2, 
                    linestyle='--', alpha=0.8)
            
            plt.title(f'{column}: Actual vs Predicted')
            plt.xlabel('Time Step')
            plt.ylabel(column)
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# save online data, needed for main program and online forecasting
def save_online_data(initial_model_path, df_online, scalers_train, context_length, df_removed_nans_forecasting, df_removed_nans_classification, variables, model_mode = model_mode):
    # Create directory if it doesn't exist
    if initial_model_path is None:
        initial_model_path = os.path.join(os.path.dirname(__file__), "forecasting_model")
    os.makedirs(initial_model_path, exist_ok=True)
    
    # Save DataFrames as CSV or Parquet to preserve structure
    df_online.to_csv(os.path.join(initial_model_path, "df_online.csv"), index=True)
    
    df_removed_nans_forecasting.to_csv(os.path.join(initial_model_path, "df_removed_nans_forecasting.csv"), index=True)
    df_removed_nans_classification.to_csv(os.path.join(initial_model_path, "df_removed_nans_classification.csv"), index=True)
    
    # Save scalers dictionary using pickle (preserves sklearn objects)
    import pickle
    with open(os.path.join(initial_model_path, "scalers_train.pkl"), 'wb') as f:
        pickle.dump(scalers_train, f)
    
    # Save simple values as numpy (these are fine as numpy)
    np.save(os.path.join(initial_model_path, "context_length.npy"), context_length)
    np.save(os.path.join(initial_model_path, "model_mode.npy"), model_mode)
    # Save variables as a text file
    with open(os.path.join(initial_model_path, "variables.txt"), 'w') as f:
        for var in variables:
            f.write(f"{var}\n")

    print(f"✓ Online data saved in original formats to: {initial_model_path}")
    print(f"  - DataFrames saved as CSV files")
    print(f"  - Scalers saved as pickle file")
    print(f"  - Variables saved as text file")
    print(f"  - Context length saved as numpy file")


def get_initial_model(initial_model_path=None, ):
    if initial_model_path is None:
        initial_model_path = os.path.join(os.path.dirname(__file__), "forecasting_model")
    
    model_file_path = os.path.join(initial_model_path, "best_model.h5")
    
    # Check if model file exists
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found at: {model_file_path}")
    
    try:
        # Try loading with custom objects to handle version compatibility
        custom_objects = {
            'mse': keras.losses.MeanSquaredError(),
            'mae': keras.metrics.MeanAbsoluteError(),
            'mean_squared_error': keras.losses.MeanSquaredError(),
            'mean_absolute_error': keras.metrics.MeanAbsoluteError(),
        }
        
        try:
            # First try loading with custom objects
            model = keras.models.load_model(model_file_path, custom_objects=custom_objects)
        except Exception as e1:
            print(f"First attempt failed: {e1}")
            try:
                # Second attempt: Load without compilation
                model = keras.models.load_model(model_file_path, compile=False)
                # Recompile the model with current Keras version
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss=keras.losses.MeanSquaredError(),
                    metrics=[keras.metrics.MeanAbsoluteError()]
                )
                print("Model loaded without compilation and recompiled successfully")
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                try:
                    # Third attempt: Rebuild model architecture and load weights only
                    print("Attempting to rebuild model architecture and load weights...")
                    
                    # Load model config to get architecture details
                    import h5py
                    with h5py.File(model_file_path, 'r') as f:
                        if 'model_config' in f.attrs:
                            import json
                            model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
                            
                            # Try to determine if it's multi-step or one-step from config
                            if 'config' in model_config and 'layers' in model_config['config']:
                                layers = model_config['config']['layers']
                                output_layer = layers[-1] if layers else None
                                
                                if output_layer and 'config' in output_layer:
                                    output_units = output_layer['config'].get('units', 0)
                                    input_shape = None
                                    
                                    # Find input shape
                                    for layer in layers:
                                        if layer.get('class_name') == 'InputLayer':
                                            batch_shape = layer.get('config', {}).get('batch_shape', [])
                                            if len(batch_shape) >= 3:
                                                context_length = batch_shape[1]
                                                num_features = batch_shape[2]
                                                input_shape = (context_length, num_features)
                                                break
                                    
                                    if input_shape:
                                        context_length, num_features = input_shape
                                        
                                        # Determine model type based on output units
                                        if output_units == num_features:
                                            # One-step model
                                            print(f"Rebuilding one-step model: input{input_shape}, output{output_units}")
                                            model = create_online_onestep_model_simple(context_length, num_features)
                                        elif output_units > num_features:
                                            # Multi-step model
                                            prediction_horizon = output_units // num_features
                                            print(f"Rebuilding multi-step model: input{input_shape}, horizon{prediction_horizon}")
                                            model = create_online_multistep_model_simple(context_length, num_features, prediction_horizon)
                                        else:
                                            raise ValueError(f"Cannot determine model type from output units: {output_units}")
                                        
                                        # Load weights
                                        model.load_weights(model_file_path)
                                        print("✅ Model architecture rebuilt and weights loaded successfully")
                                    else:
                                        raise ValueError("Could not determine input shape from model config")
                                else:
                                    raise ValueError("Could not find output layer configuration")
                            else:
                                raise ValueError("Could not parse model configuration")
                        else:
                            raise ValueError("No model configuration found in H5 file")
                            
                except Exception as e3:
                    print(f"Third attempt failed: {e3}")
                    try:
                        # Fourth attempt: Create a basic model with common architecture
                        print("Final attempt: Creating basic compatible model...")
                        
                        # Assume common settings from your training
                        context_length = 60  # Default context length
                        num_features = 23    # Based on error message batch_shape [None, 60, 23]
                        
                        # Try to create a simple multi-step model (most likely case)
                        prediction_horizon = 6  # Default from config
                        model = create_online_multistep_model_simple(context_length, num_features, prediction_horizon)
                        
                        # Try to load weights - this might work even if layer loading failed
                        try:
                            model.load_weights(model_file_path)
                            print("✅ Basic model created and weights loaded successfully")
                        except:
                            # If weights don't match, try one-step model
                            model = create_online_onestep_model_simple(context_length, num_features) 
                            model.load_weights(model_file_path)
                            print("✅ One-step model created and weights loaded successfully")
                            
                    except Exception as e4:
                        print(f"Final attempt failed: {e4}")
                        raise RuntimeError(f"All model loading attempts failed. Original error: {e1}. Consider retraining the model with current TensorFlow version.")
        
        # Basic model validation
        if model is None:
            raise ValueError("Loaded model is None")
        
        # Check if model has the expected structure
        if not hasattr(model, 'layers') or len(model.layers) == 0:
            raise ValueError("Loaded model appears to be invalid (no layers)")
        
        print(f"✓ Model loaded successfully from: {model_file_path}")
        print(f"  - Model type: {type(model)}")
        print(f"  - Number of layers: {len(model.layers)}")
        print(f"  - Input shape: {model.input_shape if hasattr(model, 'input_shape') else 'Unknown'}")
        print(f"  - Output shape: {model.output_shape if hasattr(model, 'output_shape') else 'Unknown'}")
        
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_file_path}: {str(e)}")

def get_online_data(initial_model_path):
    # Check if files exist and determine which format to use
    csv_format = os.path.exists(os.path.join(initial_model_path, "df_online.csv"))
    
    if csv_format:
        # Load DataFrames from CSV (preserves original structure)
        df_online = pd.read_csv(os.path.join(initial_model_path, "df_online.csv"), index_col=0, parse_dates=True)
        df_removed_nans_forecasting = pd.read_csv(os.path.join(initial_model_path, "df_removed_nans_forecasting.csv"), index_col=0, parse_dates=True)
        df_removed_nans_classification = pd.read_csv(os.path.join(initial_model_path, "df_removed_nans_classification.csv"), index_col=0, parse_dates=True)
        
        # Load scalers from pickle (preserves sklearn objects)
        import pickle
        with open(os.path.join(initial_model_path, "scalers_train.pkl"), 'rb') as f:
            scalers_train = pickle.load(f)
        
        # Load context length from numpy
        context_length = np.load(os.path.join(initial_model_path, "context_length.npy"), allow_pickle=True).item()
        model_mode = np.load(os.path.join(initial_model_path, "model_mode.npy"), allow_pickle=True).item()
        
        # Load variables from text file
        with open(os.path.join(initial_model_path, "variables.txt"), 'r') as f:
            variables = [line.strip() for line in f.readlines() if line.strip()]
            
        print(f"\n Online data loaded from original formats at: {initial_model_path}")
        print(f"  - DataFrames loaded from CSV files")
        print(f"  - Scalers loaded from pickle file") 
        print(f"  - Variables loaded from text file")
        
    else:
        raise FileNotFoundError(f"No data files found at: {initial_model_path}")

    return df_online, scalers_train, context_length, df_removed_nans_forecasting, df_removed_nans_classification, variables, model_mode

if __name__ == "__main__":

    ### create initial model

    ### READ DATA
    processed_forecasting_path, processed_statuses_path = get_processed_path()

    df_removed_nans_forecasting = pd.read_csv(processed_forecasting_path, index_col=0, parse_dates=True)
    df_removed_nans_classification = pd.read_csv(processed_statuses_path, index_col=0, parse_dates=True)

    original_timestamps = df_removed_nans_forecasting.index.copy()
    
    print(f"Original timestamps: {original_timestamps}")
    print(f"forecastin data index: {df_removed_nans_forecasting.index}")
    print(f"path: {processed_forecasting_path}")

    df_removed_nans_forecasting = df_removed_nans_forecasting.select_dtypes(include=[np.number])
    df_removed_nans_classification = df_removed_nans_classification.select_dtypes(include=[np.number])

    # differenciate data for forecasting

    df_differenced = df_removed_nans_forecasting.diff().dropna()

    df_initial = df_differenced.iloc[:initial_idx].copy()
    df_online = df_differenced.iloc[initial_idx:].copy()
    variables = df_initial.columns.tolist()

    if model_mode == "multi_step":
        model = create_online_multistep_model(df_initial, context_length=context_length,
                                              prediction_horizon=prediction_horizon,
                                              first_layer_units=first_layer_units,
                                              second_layer_units=second_layer_units,
                                              third_layer_units=third_layer_units,
                                              dense_units=dense_units,
                                              activation=activation,
                                              dropout_rate=dropout_rate)
        
        df_train = df_initial.iloc[:int(split_ratio * len(df_initial))]
        df_test = df_initial.iloc[int(split_ratio * len(df_initial)):]

        X_train, y_train, scalers_train, original_indices_train = split_data_for_multistep_model(df_train, context_length, prediction_horizon)
        X_test, y_test, scalers_test, original_indices_test = split_data_for_multistep_model(df_test, context_length, prediction_horizon)

        print("======================================================")
        print("Training multi-step initial model...")
        print("======================================================")

        history, model = train_model(model, X_train, y_train, epochs=epochs, use_callbacks=use_callbacks)

        # test model
            # Test the model
        df_actuals, df_predictions, all_actuals, all_predictions = test_model(model, df_initial, X_test, y_test, scalers_test, df_removed_nans_forecasting, original_indices_test, test_mode="multi_step", prediction_horizon=prediction_horizon)
        results_df, mse, mae, rmse, percentage_error, horizon_metrics = calculate_metrics(df_initial, df_actuals, df_predictions, all_actuals, all_predictions, prediction_horizon, mode="multi_step")
        plot_results(df_actuals, df_predictions, title = "Multi-Step", history = history)
        description = f"Initial Multi-Step Model Training Description:\n{model_description}\n Results:\n MSE: {mse:.6f}\n MAE: {mae:.6f}\n RMSE: {rmse:.6f}\n Percentage Error: {percentage_error:.6f}\n, Accuracy: {history.history['accuracy'][-1]:.6f}\n"


    elif model_mode == "one_step":
        model = create_online_onestep_model(df_initial, context_length=context_length,
                                            first_layer_units=first_layer_units,
                                            second_layer_units=second_layer_units,
                                            dense_units=dense_units,
                                            activation=activation,
                                            dropout_rate=dropout_rate)

        df_train = df_initial.iloc[:int(split_ratio * len(df_initial))]
        df_test = df_initial.iloc[int(split_ratio * len(df_initial)):]

        X_train, y_train, scalers_train, original_indices_train = split_data_for_onestep_model(df_train, context_length)
        X_test, y_test, scalers_test, original_indices_test = split_data_for_onestep_model(df_test, context_length)

        print("======================================================")
        print("Training one-step initial model...")
        print("======================================================")

        history, model = train_model(model, X_train, y_train, epochs=epochs, use_callbacks=use_callbacks)
        
        # test model
        df_actuals, df_predictions, all_actuals, all_predictions = test_model(model, df_initial, X_test, y_test, scalers_test, df_removed_nans_forecasting, original_indices_test, test_mode="one_step", prediction_horizon=prediction_horizon)
        results_df, mse, mae, rmse, percentage_error, horizon_metrics = calculate_metrics(df_initial, df_actuals, df_predictions, all_actuals, all_predictions, prediction_horizon, mode="one_step")
        plot_results(df_actuals, df_predictions, title = "One-Step", history = history)
        description = f"Initial One-Step Model Training Description:\n{model_description}\n Results:\n MSE: {mse:.6f}\n MAE: {mae:.6f}\n RMSE: {rmse:.6f}\n Percentage Error: {percentage_error:.6f}\n, Accuracy: {history.history['accuracy'][-1]:.6f}\n"

    # save the trained model
    model.save(os.path.join(initial_model_path, "initial_model.h5"))
    print(f"✓ Initial model saved to: {os.path.join(initial_model_path, 'initial_model.h5')}")
    
    save_online_data(initial_model_path, df_online, scalers_train, context_length, df_removed_nans_forecasting, df_removed_nans_classification, variables = variables)
    print(f"✓ Online data saved to: {initial_model_path}")

    with open(os.path.join(initial_model_path, f"{results_file_name}.txt"), "w") as f:
        f.write(description)

        

