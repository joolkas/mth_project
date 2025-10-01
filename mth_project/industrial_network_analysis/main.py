from data_utils import *
from initial_model import get_online_data, get_initial_model

from online_forecasting_one_step import one_step_rolling_buffer_learning_prediction_with_dash
from online_forecasting_multi_step import multistep_rolling_buffer_learning_prediction_with_dash
from get_data import *

from dash_plotter import *
import numpy as np
import time

import warnings
import logging

import pandas as pd

# configure warning logging
warnings_logger = logging.getLogger('warnings')
warnings_logger.setLevel(logging.WARNING)
warning_handler = logging.FileHandler('warnings.log')
warning_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
warnings_logger.addHandler(warning_handler)

# capture warnings and log them
def warning_handler_func(message, category, filename, lineno, file=None, line=None):
    warnings_logger.warning(f"{category.__name__}: {message} (File: {filename}, Line: {lineno})")

warnings.showwarning = warning_handler_func

initial_model_path = "C:\\ThesisWork\\offical_approach\\mth_project\\mth_project\\industrial_network_analysis\\forecasting_model"
online_data_path = f"{initial_model_path}"

### 1. Load data

program_options = ["predefined_data", "real_data"]
program = program_options[0]  

if program == "predefined_data":
    print("Loading online data...")
    df_online, scalers_train, context_length, df_removed_nans_forecasting, df_removed_nans_classification, variables, model_mode = get_online_data(online_data_path)
elif program == "real_data":
    print("Loading real data...")
    
### 2. Load initial model
print("Loading initial model...")
initial_model = get_initial_model(initial_model_path)
print(f"model_mode: {model_mode}")

### 3. Online forecasting with classification

prediction_horizon = 6
plotter = DashRealTimePlotter()

# Start Dash Server
print("Starting Dash server...")
plotter.start_server()

print("Open http://localhost:8050 in your browser to view real-time plots")
print("Waiting 3 seconds for server to initialize...")
time.sleep(3)

if model_mode == "one_step":
    results = one_step_rolling_buffer_learning_prediction_with_dash(initial_model,
    df_online,
    scalers_train,
    context_length,
    df_removed_nans_forecasting=df_removed_nans_forecasting,
    df_removed_nans_classification=df_removed_nans_classification,
    dash_plotter=plotter,
    variables=variables,
    prediction_horizon=prediction_horizon
    )
else:
    predictions_df, actuals_df, predictions_actuals_df, actuals_actuals_df = multistep_rolling_buffer_learning_prediction_with_dash(initial_model,
    df_online,
    scalers_train,
    context_length,
    df_removed_nans_forecasting=df_removed_nans_forecasting,
    df_removed_nans_classification=df_removed_nans_classification,
    dash_plotter=plotter,
    variables=variables,
    prediction_horizon=prediction_horizon
    )
    # Calculate errors for differenced data
    errors = predictions_df - actuals_df
    absolute_errors = errors.abs()
    
    # Basic error metrics
    mse = (errors ** 2).mean().mean()
    mae = absolute_errors.mean().mean()
    rmse = np.sqrt(mse)
    
    # SMAPE calculation
    denominator = (actuals_df.abs() + predictions_df.abs())
    # Only calculate SMAPE where denominator is not too small
    valid_mask = denominator > 1e-6
    if valid_mask.any().any():
        smape_values = (2 * absolute_errors[valid_mask] / denominator[valid_mask] * 100)
        smape = smape_values.mean().mean()
        smape_coverage = valid_mask.mean().mean()
    else:
        smape = float('inf')
        smape_coverage = 0
    
    # Directional accuracy (for non-zero values)
    nonzero_mask = (actuals_df != 0) & (predictions_df != 0)
    if nonzero_mask.any().any():
        directional_accuracy = (np.sign(actuals_df[nonzero_mask]) == np.sign(predictions_df[nonzero_mask])).mean().mean()
        direction_coverage = nonzero_mask.mean().mean()
    else:
        directional_accuracy = 0.5  # Random baseline
        direction_coverage = 0
    
    # Normalized error relative to data range
    data_range = max(actuals_df.max().max() - actuals_df.min().min(), 1e-8)
    normalized_mae = (mae / data_range * 100)
    
    print(f"Results of predictions:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    if smape != float('inf'):
        print(f"  SMAPE: {smape:.2f}% (calculated on {smape_coverage:.1%} of data)")
    else:
        print(f"  SMAPE: Not calculable (all values too small)")
    print(f"  Normalized MAE: {normalized_mae:.2f}% of data range")
    print(f"  Directional Accuracy: {directional_accuracy:.2%} (on {direction_coverage:.1%} of data)")
    print(f"  Note: Traditional MAPE not shown due to differenced data issues")
    
    # Additional diagnostic information
    print(f"\nDiagnostic Information:")
    print(f"  Data range: {actuals_df.min().min():.3f} to {actuals_df.max().max():.3f}")
    print(f"  Predictions range: {predictions_df.min().min():.3f} to {predictions_df.max().max():.3f}")
    print(f"  Mean absolute actual: {actuals_df.abs().mean().mean():.3f}")
    print(f"  Mean absolute prediction: {predictions_df.abs().mean().mean():.3f}")
    
