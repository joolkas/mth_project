import os
import csv
import pandas as pd
import numpy as np
import re
import logging

logging.basicConfig(filename='dataset.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Dataset:
    def __init__(self, filename):
        self.directory = filename
        self.read_csv()
        self._preprocessed_df = None

    def read_csv(self):
        try:
            self.df0 = pd.read_csv(self.directory, names = ['name', 'timestamp', 'value'], skiprows=1)
            logging.info("CSV file read successfully.")
            return self.df0
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            self.df0 = None  # Set to None on error
            return None

    def preprocessing(self):
        
        # avoid multiple preprocessing
        if self._preprocessed_df is not None:
            return self._preprocessed_df
            
        if self.df0 is None:
            print("Error: No data loaded. Cannot preprocess.")
            return None

        self.df0['timestamp'] = pd.to_datetime(self.df0['timestamp'])
        df_modified = self.df0.copy()
        df_modified['timestamp'] = df_modified['timestamp'].dt.floor('min').dt.strftime('%Y-%m-%d %H:%M:%S')
        reshaped_df = df_modified.pivot_table(index='timestamp', columns='name', values='value', aggfunc='first')
        reshaped_df = reshaped_df.reset_index().sort_index()
        resampled_df = reshaped_df.copy()

        # change dataframe index from rangeindex to datetimeindex, for interpolation
        resampled_df = resampled_df.set_index('timestamp')
        if not isinstance(resampled_df.index, pd.DatetimeIndex):
            resampled_df.index = pd.to_datetime(resampled_df.index)

        interpolated_df = self.smart_interpolation(resampled_df.copy())
        self._preprocessed_df = interpolated_df 
        return interpolated_df

    def smart_interpolation(self, df):
        df = df.copy()
        for column in df.columns:
            unique_count = df[column].nunique()
            total_count = len(df[column].dropna())

            if total_count == 0:
                continue

            if unique_count <= 10 and unique_count / total_count < 0.1:  # status
                df[column] = df[column].ffill()
            elif df[column].dtype in ['float64', 'int64']:  # numeric
                df[column] = df[column].interpolate(method='time')
        return df

    def print_column_names(self, name = "None", print_values = False):
        df = self.preprocessing()
        columns = df.columns.tolist()
        columns_dict = {i: col for i, col in enumerate(columns)}
        
        if name != "None":
            matching = {i: col for i, col in columns_dict.items() if name.lower() in col.lower()}
            if matching:
                columns_dict = matching
                if print_values:
                    print(f"Columns matching '{name}': \n")
                    for idx, col in matching.items():
                        print(f"{idx}: {col}")
            else:
                print(f"No columns found matching '{name}'")

    
    def get_column_names(self, name):
        df = self.preprocessing()
        if df is None:
            return []
        
        if len(name) > 0:
            matching_columns = [col for col in df.columns if name.lower() in col.lower()]
            if matching_columns:
                return matching_columns
            else:
                print(f"No columns found matching '{name}'")
                return []
        else:
            return df.columns.tolist()
        
    def get_column_values(self, column_names=[]):
        df = self.preprocessing()
        if df is None:
            print("ERROR: Preprocessing failed - no data available")
            return None
            
        if not column_names:
            return df
        else:
            try:
                missing_columns = [col for col in column_names if col not in df.columns]
                if missing_columns:
                    print(f"ERROR: Columns not found: {missing_columns}")
                    print(f"Available columns: {list(df.columns)}")
                    return None
                
                return df[column_names]
            except KeyError as e:
                print(f"Error: Column(s) not found: {e}")
                return None
            
    
    def remove_outliers(df, threshold = 3):
        if df is None:
            return None
        
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:
                mean = df[column].mean()
                std = df[column].std()
                z_scores = (df[column] - mean) / std
                clipped = df[column].copy()
                clipped[z_scores.abs() > threshold] = mean
                df[column] = clipped
        return df
    
