from data_utils import *
from initial_model import *
from online_forecasting import *
from dash_plotter import DashRealTimePlotter
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

### READ DATA
df_removed_nans_forecasting, df_removed_nans_classification = get_data()

print(f"Numeric columns: {df_removed_nans_forecasting.shape}")
print(f"Status columns: {df_removed_nans_classification.shape}")

# differenciate data for forecasting
df_differenced = df_removed_nans_forecasting.diff().dropna()

# Split data for initial training and online forecasting

initial_idx = 24 * 60 # first 24 hours for initial training
df_initial = df_differenced.iloc[:initial_idx].copy()
df_online = df_differenced.iloc[initial_idx:].copy()

### create initial model

context_length = 60
model = create_online_multivariate_model(
    df=df_initial,
    context_length=context_length,
    first_layer_units=64,
    second_layer_units=32,
    dense_units=128,
    activation='relu',
    dropout_rate=0.5
)

# split data for initial model training

split_ratio = 0.8

df_train = df_initial.iloc[:int(split_ratio * len(df_initial))]
df_test = df_initial.iloc[int(split_ratio * len(df_initial)):]

X_train, y_train, scalers_train = split_data_for_initial_model(df_train)
X_test, y_test, scalers_test = split_data_for_initial_model(df_test)

epochs = 5

print("Training initial model...")

history, initial_model = train_initial_model(model, X_train, y_train, epochs = epochs)

# test initial model
test_samples = 60

df_actuals, df_predictions = test_initial_model(model, df_initial, X_test, y_test, scalers_test, df_removed_nans_forecasting)
calculate_metrics(df_initial, df_actuals, df_predictions)
#plot_results(df_actuals, df_predictions)

print()

### online forecasting with classification

prediction_horizon = 6
plotter = DashRealTimePlotter(update_interval = 2000, prediction_horizon = prediction_horizon, port = 8050)

# Start Dash server
print("Starting Dash server...")
server_thread = plotter.start_server(debug=False, threaded=True)

print("Open http://localhost:8050 in your browser to view real-time plots")
print("Waiting 3 seconds for server to initialize...")
time.sleep(3)

variables = df_initial.columns
variables = list(variables)

# Initialize the system
forecasting_system = OnlineForecastingSystem(models_dir="C:\ThesisWork\offical_approach\mth_project\mth_project\industrial_network_analysis\classification_model")

# Run predictions
results = forecasting_system.rolling_buffer_prediction_with_dash(
    initial_model, df_online, scalers_train, context_length,
    df_removed_nans_forecasting, df_removed_nans_classification, 
    plotter, prediction_horizon=6
)
dash_stats = plotter.get_statistics()



