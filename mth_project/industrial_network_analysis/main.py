from data_preprocessing import Dataset
from data_utils import *
from initial_model import *

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
print("Reading data...")
device_name = "SW-SUPV-243"
df = Dataset(f'C:\\ThesisWork\\offical_approach\\mth_project\\mth_project\\industrial_network_analysis\\Data082025\\{device_name}.csv')

### search criteria

def get_column_names_exclude(df, search_keyword, exclude_keyword):
    df_column_names = df.get_column_names(search_keyword)
    df_column_names_excluded = []

    for col in df_column_names:
        if exclude_keyword.lower() not in col.lower():
            df_column_names_excluded.append(col)

    return df_column_names_excluded

# names for STATUSES
statuses = ': operational status'
statuses_exclude = 'Unused'

# names for OTHER NUMERIC PARAMETERS
icmp_params = 'ICMP'
temperature = 'temperature'
cpu = 'cpu'
memory = 'used memory'

numeric_exclude = "status"

# names for traffic on ports - BITS SENT/RECEIVED
bits = "bits"
bits_exclude = "unused"

df_numerics = get_column_names_exclude(df, icmp_params, numeric_exclude)\
     + get_column_names_exclude(df, temperature, numeric_exclude)\
     + get_column_names_exclude(df, cpu, numeric_exclude)\
     + get_column_names_exclude(df, memory, numeric_exclude)\
     + get_column_names_exclude(df, bits, bits_exclude)

df_statuses = get_column_names_exclude(df, statuses, statuses_exclude)

print("Getting values...")
df_numeric_values = df.get_column_values(df_numerics)
df_status_values = df.get_column_values(df_statuses)

print(f"Numeric columns: {df_numeric_values.shape}")
print(f"Status columns: {df_status_values.shape}")

# limit amount of values, use numeric for forecasting
df_all_program = pd.concat([df_numeric_values, df_status_values], axis=1)
df_forecasting = df_numeric_values.iloc[8000:]

# PREPROCESSING
print("Preprocessing data...")

df_removed_outliers = remove_outliers(df_forecasting, 3)
df_removed_nans = df_removed_outliers.dropna(axis=1, how="all")

# Split data for initial training and online forecasting

initial_idx = 24 * 60 # first 24 hours for initial training
df_initial = df_removed_nans.iloc[:initial_idx].copy()
df_online = df_removed_nans.iloc[initial_idx:].copy()

# Create online model

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

X_train, y_train, scalers = split_data_for_initial_model(df_train)
X_test, y_test, scalers = split_data_for_initial_model(df_test)

epochs = 5

print("Training initial model...")

history, initial_model = train_initial_model(model, X_train, y_train, epochs = epochs)

# test model
test_samples = 60

df_actuals, df_predictions = test_initial_model(model, df_initial, X_test, y_test, scalers)
calculate_metrics(df_initial, df_actuals, df_predictions)
plot_results(df_actuals, df_predictions)

print()

