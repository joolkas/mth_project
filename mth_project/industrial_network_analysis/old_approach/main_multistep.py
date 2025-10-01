from data_utils import *
from initial_model_multi_step import get_multistep_online_data, get_multistep_model
from online_forecasting_learning_multistep import rolling_buffer_multistep_learning_prediction_with_dash

from get_data import *

from dash_plotter import *
import numpy as np
import time

import warnings
import logging

import pandas as pd

# Configure warning logging
warnings_logger = logging.getLogger('warnings')
warnings_logger.setLevel(logging.WARNING)
warning_handler = logging.FileHandler('warnings.log')
warning_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
warnings_logger.addHandler(warning_handler)

# Capture warnings and log them
def warning_handler_func(message, category, filename, lineno, file=None, line=None):
    warnings_logger.warning(f"{category.__name__}: {message} (File: {filename}, Line: {lineno})")

warnings.showwarning = warning_handler_func

# Multi-step model paths
initial_model_path = "C:\\ThesisWork\\offical_approach\\mth_project\\mth_project\\industrial_network_analysis\\forecasting_model"
online_data_path = f"{initial_model_path}\\online_data_multistep"
print(f"Multi-step online data path: {online_data_path}")
print(f"Multi-step initial model path: {initial_model_path}")

### 1. Load data

program_options = ["predefined_data", "real_data"]
program = program_options[0]  

if program == "predefined_data":
    print("Loading multi-step online data...")
    try:
        df_online, scalers_train, context_length, df_removed_nans_forecasting, df_removed_nans_classification, variables, prediction_horizon = get_multistep_online_data(online_data_path)
        print(f"✓ Multi-step data loaded successfully")
        print(f"  - Data shape: {df_online.shape}")
        print(f"  - Prediction horizon: {prediction_horizon}")
        print(f"  - Context length: {context_length}")
        print(f"  - Variables: {len(variables)}")
    except Exception as e:
        print(f"Error loading multi-step data: {e}")
        print("Please run initial_model_multi_step.py first to generate multi-step training data and model.")
        exit(1)
        
elif program == "real_data":
    print("Loading real data...")
    # This would be implemented for real-time data integration

### 2. Load multi-step initial model
print("Loading multi-step initial model...")
try:
    initial_model = get_multistep_model(initial_model_path)
    print(f"✓ Multi-step model loaded successfully")
    print(f"  - Input shape: {initial_model.input_shape}")
    print(f"  - Output shape: {initial_model.output_shape}")
    print(f"  - Expected output features: {initial_model.output_shape[1]} (should be {prediction_horizon * len(variables)})")
    
    # Validate model output shape
    expected_output_size = prediction_horizon * len(variables)
    if initial_model.output_shape[1] != expected_output_size:
        print(f"⚠ Warning: Model output shape mismatch!")
        print(f"  Expected: {expected_output_size} (horizon={prediction_horizon} × features={len(variables)})")
        print(f"  Actual: {initial_model.output_shape[1]}")
        print("  The model may not work correctly. Please retrain with matching parameters.")
    
except Exception as e:
    print(f"Error loading multi-step model: {e}")
    print("Please run initial_model_multi_step.py first to train the multi-step model.")
    exit(1)

### 3. Multi-step online forecasting with classification

plotter = DashRealTimePlotter()

# Start Dash server
print("Starting Dash server for multi-step predictions...")
server_thread = plotter.start_server()

print("Open http://localhost:8050 in your browser to view real-time multi-step plots")
print("Waiting 3 seconds for server to initialize...")
time.sleep(3)

# Run multi-step predictions
print("\n" + "="*60)
print("STARTING MULTI-STEP ONLINE LEARNING FORECASTING")
print("="*60)
print(f"Prediction method: ensemble (multi-step)")
print(f"Prediction horizon: {prediction_horizon} steps")
print(f"Context length: {context_length} timesteps")
print(f"Variables: {len(variables)}")
print("Press 'Q' to quit during execution")
print("="*60)

try:
    results = rolling_buffer_multistep_learning_prediction_with_dash(
        initial_model=initial_model,
        df_online=df_online, 
        scalers=scalers_train, 
        context_length=context_length,
        df_removed_nans_forecasting=df_removed_nans_forecasting,
        df_removed_nans_classification=df_removed_nans_classification, 
        dash_plotter=plotter,
        variables=variables, 
        prediction_horizon=prediction_horizon,
        prediction_method='ensemble'  # Can be 'direct', 'regularized', or 'ensemble'
    )
    
    predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df = results
    
    print("\n" + "="*60)
    print("MULTI-STEP FORECASTING COMPLETED")
    print("="*60)
    print(f"Total predictions made: {len(predictions_df)}")
    print(f"Final prediction shape: {predictions_df.shape}")
    print(f"Final actuals shape: {actuals_df.shape}")
    
    # Calculate final performance metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    overall_mae = mean_absolute_error(actuals_df.values, predictions_df.values)
    overall_mse = mean_squared_error(actuals_df.values, predictions_df.values)
    overall_rmse = np.sqrt(overall_mse)
    
    print(f"\n📊 FINAL MULTI-STEP PERFORMANCE METRICS:")
    print(f"  - Mean Absolute Error (MAE): {overall_mae:.6f}")
    print(f"  - Mean Squared Error (MSE): {overall_mse:.6f}")
    print(f"  - Root Mean Squared Error (RMSE): {overall_rmse:.6f}")
    
    # Per-variable performance
    print(f"\n📈 PER-VARIABLE PERFORMANCE:")
    for i, var in enumerate(variables):
        var_mae = mean_absolute_error(actuals_df.iloc[:, i], predictions_df.iloc[:, i])
        print(f"  - {var}: MAE = {var_mae:.6f}")
    
    dash_stats = plotter.get_statistics()
    print(f"\n📱 DASHBOARD STATISTICS:")
    for key, value in dash_stats.items():
        print(f"  - {key}: {value}")

except KeyboardInterrupt:
    print("\n⚠ Program interrupted by user")
except Exception as e:
    print(f"\n❌ Error during multi-step forecasting: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Multi-step forecasting program finished.")