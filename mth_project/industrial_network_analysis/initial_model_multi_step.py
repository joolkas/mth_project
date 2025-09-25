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

# Multi-step model parameters
epochs = 80
batch_size = 64
validation_split = 0.2
verbose = 1
context_length = 60    # Input window length
prediction_horizon = 6  # Number of future steps to predict directly
first_layer_units = 256
second_layer_units = 128
third_layer_units = 64  # Additional layer for multi-step complexity
dense_units = 512       # Larger dense layer for multi-step output
activation = 'relu'
dropout_rate = 0.3

use_callbacks = True

# Split data for initial training and online forecasting
initial_idx = 48 * 60 # first 48 hours for initial training

model_description = f"Multi-Step Model - Epochs: {epochs}, Batch Size: {batch_size}, Validation Split: {validation_split}, Context Length: {context_length}, Prediction Horizon: {prediction_horizon}, First Layer Units: {first_layer_units}, Second Layer Units: {second_layer_units}, Third Layer Units: {third_layer_units}, Dense Units: {dense_units}, Activation: {activation}, Dropout Rate: {dropout_rate}"
results_file_name = "multi_step_model_results_48h_001"

def create_online_multistep_model(df,
                                  context_length=context_length,
                                  prediction_horizon=prediction_horizon,
                                  first_layer_units=first_layer_units,
                                  second_layer_units=second_layer_units,
                                  third_layer_units=third_layer_units,
                                  dense_units=dense_units,
                                  activation=activation,
                                  dropout_rate=dropout_rate):
    """
    Create a multi-step multivariate model optimized for direct multi-horizon prediction.
    
    Args:
        df: DataFrame containing the data
        context_length: Length of the input context window
        prediction_horizon: Number of future timesteps to predict directly
        first_layer_units: Number of units in the first LSTM layer
        second_layer_units: Number of units in the second LSTM layer
        third_layer_units: Number of units in the third LSTM layer
        dense_units: Number of units in the dense layer
        activation: Activation function
        dropout_rate: Dropout rate

    Returns:
        model: Compiled Keras model
    """
    num_features = len(df.columns)
    
    # Multi-step output: prediction_horizon * num_features
    output_size = prediction_horizon * num_features
    
    model = keras.models.Sequential([
        # First LSTM layer with return sequences for deeper processing
        keras.layers.LSTM(first_layer_units, return_sequences=True, 
                         input_shape=(context_length, num_features)),
        keras.layers.Dropout(dropout_rate * 0.5),  # Lighter dropout in LSTM layers
        
        # Second LSTM layer with return sequences for additional depth
        keras.layers.LSTM(second_layer_units, return_sequences=True),
        keras.layers.Dropout(dropout_rate * 0.5),
        
        # Third LSTM layer without return sequences (final encoding)
        keras.layers.LSTM(third_layer_units, return_sequences=False),
        keras.layers.Dropout(dropout_rate * 0.7),
        
        # Dense layers for multi-step output processing
        keras.layers.Dense(dense_units, activation=activation),
        keras.layers.Dropout(dropout_rate),
        
        # Intermediate dense layer for complexity
        keras.layers.Dense(dense_units // 2, activation=activation),
        keras.layers.Dropout(dropout_rate * 0.5),
        
        # Final output layer: outputs all future timesteps at once
        keras.layers.Dense(output_size, activation='linear')  # Linear for regression
    ])
    
    # Custom loss function that can weight different prediction horizons
    def weighted_mse_loss(y_true, y_pred):
        """
        Weighted MSE loss that gives more importance to near-term predictions
        """
        # Reshape predictions to (batch_size, prediction_horizon, num_features)
        y_true_reshaped = keras.ops.reshape(y_true, (-1, prediction_horizon, num_features))
        y_pred_reshaped = keras.ops.reshape(y_pred, (-1, prediction_horizon, num_features))
        
        # Create weights that decrease with prediction distance
        # Give more weight to near-term predictions (t+1, t+2) vs far-term (t+5, t+6)
        horizon_weights = keras.ops.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])  # Adjust as needed
        horizon_weights = horizon_weights[:prediction_horizon]  # Ensure correct length
        
        # Calculate squared errors for each timestep
        squared_errors = keras.ops.square(y_true_reshaped - y_pred_reshaped)
        
        # Apply weights across the prediction horizon dimension
        weighted_errors = squared_errors * horizon_weights[None, :, None]
        
        # Return mean weighted error
        return keras.ops.mean(weighted_errors)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001), 
        loss=weighted_mse_loss,  # Use weighted loss
        metrics=['mae']
    )
    
    return model

def split_data_for_multistep_model(df, context_length=context_length, prediction_horizon=prediction_horizon):
    """
    Create training data for multi-step prediction model.
    
    Returns:
        X_train: Input sequences of shape (samples, context_length, features)
        y_train: Multi-step targets of shape (samples, prediction_horizon * features)
        scalers: Dictionary of scalers for each variable
    """
    # Create scalers dictionary to store individual scalers for each variable
    scalers = {}
    scaled_data = np.zeros_like(df.values)

    # Scale each column separately
    for i, var in enumerate(df.columns):
        scaler = StandardScaler()
        scaled_data[:, i] = scaler.fit_transform(df[var].values.reshape(-1, 1)).flatten()
        # Store the scaler for inverse transform
        scalers[var] = scaler  

    # Convert back to DataFrame with original column names and index
    df_scaled = pd.DataFrame(
        data=scaled_data,
        columns=df.columns,
        index=df.index
    )

    # Create training sequences for multi-step prediction
    X_train, y_train = [], []

    # We need context_length + prediction_horizon data points to create one sample
    for i in range(context_length, len(df_scaled) - prediction_horizon + 1):
        # Input: context_length timesteps
        X_train.append(df_scaled.iloc[i-context_length:i].values)
        
        # Output: next prediction_horizon timesteps (flattened)
        future_steps = []
        for step in range(prediction_horizon):
            future_steps.extend(df_scaled.iloc[i + step].values)
        y_train.append(future_steps)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Multi-step training data shapes:")
    print(f"  X_train: {X_train.shape} (samples, context_length, features)")
    print(f"  y_train: {y_train.shape} (samples, prediction_horizon * features)")

    return X_train, y_train, scalers

def train_multistep_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size, 
                         validation_split=validation_split, verbose=verbose, use_callbacks=False):
    """
    Train the multi-step prediction model with enhanced callbacks.
    """
    callback_list = []
    
    if use_callbacks:
        callback_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=8,  # More patience for complex multi-step model
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,  # Less aggressive learning rate reduction
                patience=4,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'C:\\ThesisWork\\offical_approach\\mth_project\\mth_project\\industrial_network_analysis\\forecasting_model\\best_multistep_model.h5',
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

def predict_multistep_direct(model, context, variables, prediction_horizon=prediction_horizon):
    """
    Make direct multi-step predictions using the trained multi-step model.
    
    Args:
        model: Trained multi-step model
        context: Input context of shape (context_length, num_features)
        variables: List of variable names
        prediction_horizon: Number of steps to predict
    
    Returns:
        predictions: List of prediction arrays, one for each future timestep
    """
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

def filter_status_predictions(predictions_df, column_names, threshold=0.5):
    """Apply rounding to status columns in predictions."""
    filtered_df = predictions_df.copy()
    
    for col in column_names:
        if any(keyword in col.lower() for keyword in ['status', 'operational', 'interface']):
            filtered_df[col] = np.round(filtered_df[col])
    
    return filtered_df

def inverse_difference(predictions_arrays, last_actual_values):
    """Convert differenced predictions back to actual values."""
    actual_predictions = []
    current_values = last_actual_values.copy()
    
    for pred_diff in predictions_arrays:
        # Add difference to get actual value
        current_values = current_values + np.array(pred_diff)
        actual_predictions.append(current_values.copy())
    
    return actual_predictions

def test_multistep_model(model, df, X_test, y_test, scalers, df_removed_nans_forecasting, 
                        prediction_horizon=prediction_horizon):
    """
    Test the multi-step model and convert predictions back to original scale.
    """
    predictions_test = []
    actuals_test = []
    
    n_features = len(df.columns)

    for i in range(len(y_test)):
        # Get context for prediction
        context = X_test[i]
        
        # Make multi-step prediction
        step_predictions = predict_multistep_direct(model, context, df.columns, prediction_horizon)
        
        # Convert y_test back to multi-step format for comparison
        actual_steps = []
        for step in range(prediction_horizon):
            start_idx = step * n_features
            end_idx = (step + 1) * n_features
            actual_steps.append(y_test[i][start_idx:end_idx])
        
        predictions_test.append(step_predictions)
        actuals_test.append(actual_steps)

    # Convert to arrays and inverse transform
    all_predictions = []
    all_actuals = []
    
    for sample_idx in range(len(predictions_test)):
        sample_preds = []
        sample_actuals = []
        
        for step in range(prediction_horizon):
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
    forecasting_variables = df.columns.tolist()
    last_actual_values = df_removed_nans_forecasting[forecasting_variables].mean().values

    predictions_original = inverse_difference(first_step_predictions, last_actual_values)
    actuals_original = inverse_difference(first_step_actuals, last_actual_values)

    # Convert to DataFrames
    actuals_df = pd.DataFrame(data=actuals_original, columns=df.columns)
    predictions_df = pd.DataFrame(data=predictions_original, columns=df.columns)

    return actuals_df, filter_status_predictions(predictions_df, df.columns), all_predictions, all_actuals

def calculate_multistep_metrics(df, actuals_original, predictions_original, all_actuals, all_predictions, 
                               prediction_horizon=prediction_horizon):
    """
    Calculate metrics for multi-step predictions including horizon-specific performance.
    """
    # Overall metrics (using first step)
    mse = mean_squared_error(actuals_original, predictions_original)
    mae = mean_absolute_error(actuals_original, predictions_original)
    rmse = np.sqrt(mse)
    percentage_error = np.mean(np.abs((actuals_original - predictions_original) / actuals_original)) * 100

    print(f"\nMulti-Step Model Performance (t+1 predictions):")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Percentage Error: {percentage_error:.6f}")

    # Calculate metrics for each prediction horizon
    horizon_metrics = []
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
    
    return results_df, mse, mae, rmse, percentage_error, horizon_metrics

def plot_multistep_results(actuals_df, predictions_df, history=None, horizon_metrics=None):
    """
    Plot results for multi-step model including training history and horizon performance.
    """
    if history is not None:
        # Plot training history
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Multi-Step Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Multi-Step Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        # Plot horizon-specific performance if available
        plt.subplot(1, 3, 3)
        if horizon_metrics is not None:
            horizons = [m['horizon'] for m in horizon_metrics]
            maes = [m['mae'] for m in horizon_metrics]
            mses = [m['mse'] for m in horizon_metrics]
            
            plt.plot(horizons, maes, 'o-', label='MAE by Horizon', color='blue')
            plt.xlabel('Prediction Horizon (t+n)')
            plt.ylabel('MAE')
            plt.title('Performance by Prediction Horizon')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Horizon metrics\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.show()

    # Plot predictions vs actuals for each variable
    n_columns = len(actuals_df.columns)
    fig_width = min(20, max(12, n_columns * 2))
    fig_height = n_columns * 4
    
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
        
        plt.title(f'{column}: Multi-Step Actual vs Predicted (t+1)')
        plt.xlabel('Time Step')
        plt.ylabel(column)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def save_multistep_online_data(initial_model_path, df_online, scalers_train, context_length, 
                              df_removed_nans_forecasting, df_removed_nans_classification, 
                              variables, prediction_horizon):
    """
    Save online data for multi-step model including prediction horizon information.
    """
    # Create directory if it doesn't exist
    os.makedirs(initial_model_path, exist_ok=True)
    
    # Save DataFrames as CSV
    df_online.to_csv(f"{initial_model_path}\\df_online.csv", index=True)
    df_removed_nans_forecasting.to_csv(f"{initial_model_path}\\df_removed_nans_forecasting.csv", index=True)
    df_removed_nans_classification.to_csv(f"{initial_model_path}\\df_removed_nans_classification.csv", index=True)
    
    # Save scalers dictionary using pickle
    with open(f"{initial_model_path}\\scalers_train.pkl", 'wb') as f:
        pickle.dump(scalers_train, f)
    
    # Save model parameters
    np.save(f"{initial_model_path}\\context_length.npy", context_length)
    np.save(f"{initial_model_path}\\prediction_horizon.npy", prediction_horizon)
    
    # Save variables as text file
    variables_list = list(variables) if hasattr(variables, 'tolist') else list(variables)
    with open(f"{initial_model_path}\\variables.txt", 'w') as f:
        for var in variables_list:
            f.write(f"{var}\n")
    
    print(f"✓ Multi-step online data saved to: {initial_model_path}")
    print(f"  - Prediction horizon: {prediction_horizon}")

def get_multistep_online_data(initial_model_path):
    """
    Load online data for multi-step model in their original formats.
    
    Args:
        initial_model_path: Path to load the data from
        
    Returns:
        tuple: (df_online, scalers_train, context_length, df_removed_nans_forecasting, 
                df_removed_nans_classification, variables, prediction_horizon)
    """
    
    # Check if files exist
    csv_format = os.path.exists(f"{initial_model_path}\\df_online.csv")
    
    if csv_format:
        # Load DataFrames from CSV
        df_online = pd.read_csv(f"{initial_model_path}\\df_online.csv", index_col=0, parse_dates=True)
        df_removed_nans_forecasting = pd.read_csv(f"{initial_model_path}\\df_removed_nans_forecasting.csv", index_col=0, parse_dates=True)
        df_removed_nans_classification = pd.read_csv(f"{initial_model_path}\\df_removed_nans_classification.csv", index_col=0, parse_dates=True)
        
        # Load scalers from pickle
        import pickle
        with open(f"{initial_model_path}\\scalers_train.pkl", 'rb') as f:
            scalers_train = pickle.load(f)
        
        # Load parameters from numpy
        context_length = np.load(f"{initial_model_path}\\context_length.npy", allow_pickle=True).item()
        prediction_horizon = np.load(f"{initial_model_path}\\prediction_horizon.npy", allow_pickle=True).item()
        
        # Load variables from text file
        with open(f"{initial_model_path}\\variables.txt", 'r') as f:
            variables = [line.strip() for line in f.readlines() if line.strip()]
            
        print(f"\n✓ Multi-step online data loaded from: {initial_model_path}")
        print(f"  - Prediction horizon: {prediction_horizon}")
        print(f"  - Context length: {context_length}")
        print(f"  - Variables: {len(variables)}")
        
    else:
        raise FileNotFoundError(f"No multi-step data files found at: {initial_model_path}")

    return df_online, scalers_train, context_length, df_removed_nans_forecasting, df_removed_nans_classification, variables, prediction_horizon

def get_multistep_model(initial_model_path=None):
    """
    Load the trained multi-step model from the specified path.
    """
    if initial_model_path is None:
        initial_model_path = "C:\\ThesisWork\\offical_approach\\mth_project\\mth_project\\industrial_network_analysis\\forecasting_model"
    
    model_file_path = f"{initial_model_path}\\multistep_model.h5"
    
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Multi-step model file not found at: {model_file_path}")
    
    try:
        # Try loading with compilation disabled first, then recompile
        model = keras.models.load_model(model_file_path, compile=False)
        
        # Recompile with the same weighted loss function
        def weighted_mse_loss(y_true, y_pred):
            # Recreate the weighted loss function
            prediction_horizon = 6  # Default, should match training
            num_features = y_pred.shape[1] // prediction_horizon
            
            y_true_reshaped = keras.ops.reshape(y_true, (-1, prediction_horizon, num_features))
            y_pred_reshaped = keras.ops.reshape(y_pred, (-1, prediction_horizon, num_features))
            
            horizon_weights = keras.ops.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
            horizon_weights = horizon_weights[:prediction_horizon]
            
            squared_errors = keras.ops.square(y_true_reshaped - y_pred_reshaped)
            weighted_errors = squared_errors * horizon_weights[None, :, None]
            
            return keras.ops.mean(weighted_errors)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=weighted_mse_loss,
            metrics=['mae']
        )
        
        print(f"✓ Multi-step model loaded and recompiled successfully from: {model_file_path}")
        print(f"  - Model type: {type(model)}")
        print(f"  - Number of layers: {len(model.layers)}")
        print(f"  - Input shape: {model.input_shape}")
        print(f"  - Output shape: {model.output_shape}")
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load multi-step model from {model_file_path}: {str(e)}")

if __name__ == "__main__":
    ### Create multi-step initial model

    ### READ DATA
    processed_forecasting_path, processed_statuses_path = get_processed_path()

    df_removed_nans_forecasting = pd.read_csv(processed_forecasting_path, index_col=0, parse_dates=True)
    df_removed_nans_classification = pd.read_csv(processed_statuses_path, index_col=0, parse_dates=True)

    original_timestamps = df_removed_nans_forecasting.index.copy()
    
    print(f"Multi-step model training started...")
    print(f"Prediction horizon: {prediction_horizon} steps")

    df_removed_nans_forecasting = df_removed_nans_forecasting.select_dtypes(include=[np.number])
    df_removed_nans_classification = df_removed_nans_classification.select_dtypes(include=[np.number])

    # Differentiate data for forecasting
    df_differenced = df_removed_nans_forecasting.diff().dropna()

    df_initial = df_differenced.iloc[:initial_idx].copy()
    df_online = df_differenced.iloc[initial_idx:].copy()

    variables = df_initial.columns

    # Create multi-step model
    model = create_online_multistep_model(
        df=df_initial,
        context_length=context_length,
        prediction_horizon=prediction_horizon,
        first_layer_units=first_layer_units,
        second_layer_units=second_layer_units,
        third_layer_units=third_layer_units,
        dense_units=dense_units,
        activation=activation,
        dropout_rate=dropout_rate
    )

    # Split data for model training
    split_ratio = 0.8

    df_train = df_initial.iloc[:int(split_ratio * len(df_initial))]
    df_test = df_initial.iloc[int(split_ratio * len(df_initial)):]

    X_train, y_train, scalers_train = split_data_for_multistep_model(df_train, context_length, prediction_horizon)
    X_test, y_test, scalers_test = split_data_for_multistep_model(df_test, context_length, prediction_horizon)

    print("======================================================")
    print("Training multi-step initial model...")
    print("======================================================")

    history, multistep_model = train_multistep_model(model, X_train, y_train, epochs=epochs, use_callbacks=use_callbacks)

    # Save multi-step model
    initial_model_path = "C:\\ThesisWork\\offical_approach\\mth_project\\mth_project\\industrial_network_analysis\\forecasting_model"
    multistep_model.save(f"{initial_model_path}\\multistep_model.h5")

    # Test the model
    df_actuals, df_predictions, all_predictions, all_actuals = test_multistep_model(
        multistep_model, df_initial, X_test, y_test, scalers_test, df_removed_nans_forecasting, prediction_horizon)
    
    results_df, mse, mae, rmse, percentage_error, horizon_metrics = calculate_multistep_metrics(
        df_initial, df_actuals, df_predictions, all_actuals, all_predictions, prediction_horizon)
    
    # Save online data for multi-step approach
    save_multistep_online_data(f"{initial_model_path}\\online_data_multistep", df_online, scalers_train, 
                              context_length, df_removed_nans_forecasting, df_removed_nans_classification, 
                              variables, prediction_horizon)
    
    # Plot results
    plot_multistep_results(df_actuals, df_predictions, history, horizon_metrics)

    # Save description
    description = f"Multi-Step Model Training Description:\n{model_description}\n Results:\n MSE: {mse:.6f}\n MAE: {mae:.6f}\n RMSE: {rmse:.6f}\n Percentage Error: {percentage_error:.6f}\n"
    
    # Add horizon-specific results
    description += "\nHorizon-specific performance:\n"
    for metric in horizon_metrics:
        description += f"  t+{metric['horizon']}: MSE={metric['mse']:.6f}, MAE={metric['mae']:.6f}\n"
    
    with open(f"{initial_model_path}\\{results_file_name}.txt", "w") as f:
        f.write(description)
        
    print(f"\n✓ Multi-step model training completed!")
    print(f"✓ Model saved as: multistep_model.h5")
    print(f"✓ Results saved as: {results_file_name}.txt")