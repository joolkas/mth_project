import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler

def remove_outliers(df, threshold=3):
    if df is None:
        print("ERROR: remove_outliers received None as input")
        return None
    
    if not isinstance(df, pd.DataFrame):
        print(f"ERROR: remove_outliers expected DataFrame, got {type(df)}")
        return None
    
    # create a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype in ['float64', 'int64']:
            mean = df_cleaned[column].mean()
            std = df_cleaned[column].std()
            
            # skip if std is 0 (constant column)
            if std == 0:
                continue
                
            z_scores = (df_cleaned[column] - mean) / std
            clipped = df_cleaned[column].copy()
            clipped[z_scores.abs() > threshold] = mean
            df_cleaned[column] = clipped
    
    return df_cleaned

def normalize(df, method='robust'):
    if method == "minmax":
        return (df - df.min()) / (df.max() - df.min())
    elif method == "zscore":
        return (df - df.mean()) / df.std()
    elif method == "robust":
        scaler = RobustScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    else:
        raise ValueError("Unknown normalization method")