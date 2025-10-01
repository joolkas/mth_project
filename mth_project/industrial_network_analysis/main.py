from data_utils import *
from initial_model import get_online_data, get_initial_model

from online_forecasting import rolling_buffer_learning_prediction_with_dash

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
online_data_path = f"{initial_model_path}\\online_data"

### 1. Load data

program_options = ["predefined_data", "real_data"]
program = program_options[0]  

if program == "predefined_data":
    print("Loading online data...")
    df_online, scalers_train, context_length, df_removed_nans_forecasting, df_removed_nans_classification, variables = get_online_data(online_data_path)
elif program == "real_data":
    print("Loading real data...")
    

### 2. Load initial model
print("Loading initial model...")
initial_model = get_initial_model(initial_model_path)
