from data_preprocessing import Dataset
from data_utils import *
from initial_model import *
from online_forecasting import *

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

print("Reading data...")

### search criteria

def get_column_names_exclude(df, search_keyword, exclude_keyword):
    df_column_names = df.get_column_names(search_keyword)
    df_column_names_excluded = []

    for col in df_column_names:
        if exclude_keyword.lower() not in col.lower():
            df_column_names_excluded.append(col)

    return df_column_names_excluded

def get_data(device_name="SW-SUPV-243", data_path="C:\\ThesisWork\\offical_approach\\mth_project\\mth_project\\industrial_network_analysis\\Data082025\\",\
              numeric_names=["ICMP", "temperature", "cpu", "used memory", "bits"], status_names = [": operational status"], numeric_exclude ="status", status_exclude="unused"):
    
    df = Dataset(f'{data_path}{device_name}.csv')

    df_numerics = []
    df_statuses = []

    for numeric_name in numeric_names:
        df_numerics += get_column_names_exclude(df, numeric_name, numeric_exclude)
    
    for status_name in status_names:
        df_statuses += get_column_names_exclude(df, status_name, status_exclude)


    print("Getting values...")
    df_numeric_values = df.get_column_values(df_numerics)
    df_status_values = df.get_column_values(df_statuses)

    print(f"Numeric columns: {df_numeric_values.shape}")
    print(f"Status columns: {df_status_values.shape}")

    # limit amount of values, use numeric for forecasting

    df_forecasting = df_numeric_values.iloc[8000:]
    df_classification = df_status_values.iloc[8000:]

    # PREPROCESSING
    print("Preprocessing data...")

    # finish data preprocessing for forecasting
    df_removed_outliers_forecasting = remove_outliers(df_forecasting, 1000)
    df_removed_nans_forecasting = df_removed_outliers_forecasting.dropna(axis=1, how="all")

    # preprocessing for classification, which uses all data
    df_removed_outliers_statuses = remove_outliers(df_classification, 1000)
    df_removed_nans_statuses = df_removed_outliers_statuses.dropna(axis=1, how="all")

    return df_removed_nans_forecasting, df_removed_nans_statuses