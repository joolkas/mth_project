from data_preprocessing import Dataset

import warnings
import logging

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
device_name = "A1-RBX-242"
df = Dataset(f'mth_project/industrial_network_analysis/Data082025/{device_name}.csv')

# search criteria
statuses = ': operational status'
statuses_exclude = 'Unused'

icmp_params = 'ICMP'
temperature = 'temperature'

numeric_exclude = "status"

# create name list for statuses
df_statuses_column_names = df.get_column_names(statuses)
df_statuses_column_names_excluded = []

for col in df_statuses_column_names:
    if statuses_exclude not in col:
        df_statuses_column_names_excluded.append(col)

# create name list for numeric values
df_ICMP_column_names = df.get_column_names(icmp_params)
df_temperature_names = df.get_column_names(temperature)

df_numeric = df_ICMP_column_names + df_temperature_names
df_numeric_exclude = []

for col in df_numeric:
    if numeric_exclude not in col:
        df_numeric_exclude.append(col)

# combine column names
df_result_column_names = df_statuses_column_names_excluded + df_numeric_exclude

# get values of defines columns
df0 = df.get_column_values(df_result_column_names)
df0 = df0.iloc[8000:]


