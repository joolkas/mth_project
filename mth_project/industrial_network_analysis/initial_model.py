from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import time
import sys
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

def create_online_multivariate_model(df,
                                     context_length=60,
                                     first_layer_units=64,
                                     second_layer_units=64,
                                     dense_units=128,
                                     activation='relu',
                                     dropout_rate=0.5):
    """
    Create a multivariate model optimized for online learning.
    Clean version for Online Approach 3.
    
    Args:
        df: DataFrame containing the data
        context_length: Length of the context window
        first_layer_units: Number of units in the first LSTM layer
        second_layer_units: Number of units in the second LSTM layer
        dense_units: Number of units in the dense layer
        activation: Activation function
        dropout_rate: Dropout rate

    Returns:
        model: Compiled Keras model
    """
    num_features = len(df.columns)
    
    model = keras.models.Sequential([
        keras.layers.LSTM(first_layer_units, return_sequences=True, input_shape=(context_length, num_features)),
        keras.layers.LSTM(second_layer_units, return_sequences=False),
        keras.layers.Dense(dense_units, activation=activation),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(len(df.columns))  # Output for each target variable
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    
    return model


def split_data_for_initial_model(df, context_length=60):
    # create scalers dictionary to store individual scalers for each variable
    scalers = {}
    scaled_data = np.zeros_like(df.values)

    # scale each column separately
    for i, var in enumerate(df.columns):
        scaler = StandardScaler()
        scaled_data[:, i] = scaler.fit_transform(df[var].values.reshape(-1, 1)).flatten()
        # store the scaler for inverse transform
        scalers[var] = scaler  

    # convert back to DataFrame with original column names and index
    df_initial_scaled = pd.DataFrame(
        data=scaled_data,
        columns=df.columns,
        index=df.index
    )

    # divide data to train and test sets

    X_train, y_train = [], []

    for i in range(context_length, len(df_initial_scaled)):
        X_train.append(df_initial_scaled.iloc[i-context_length:i].values)
        y_train.append(df_initial_scaled.iloc[i].values)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train, scalers

def train_initial_model(model, X_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.2, verbose =  1):

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose
    )

    return history, model

def filter_status_predictions(predictions_df, column_names, threshold=0.5):
    filtered_df = predictions_df.copy()
    
    for col in column_names:
        if any(keyword in col.lower() for keyword in ['status', 'operational', 'interface']):
            filtered_df[col] = np.round(filtered_df[col])
    
    return filtered_df

def inverse_difference(predictions_arrays, last_actual_values):
    actual_predictions = []
    current_values = last_actual_values.copy()
    
    for pred_diff in predictions_arrays:
        # Add difference to get actual value
        current_values = current_values + np.array(pred_diff)
        actual_predictions.append(current_values.copy())
    
    return actual_predictions

def test_initial_model(model, df, X_test, y_test, scalers, df_removed_nans_forecasting):
    predictions_test = []
    actuals_test = []

    for i in range(len(y_test)):
        context = X_test[i].reshape(1, X_test.shape[1], len(df.columns))
        prediction = model.predict(context, verbose=0)
        
        # without flatten:
        # Output: [[0.123, 0.456]] - 2D array with batch dimension

        # with flatten:
        # output: [0.123, 0.456] - clean 1D array

        predictions_test.append(prediction.flatten())
        actuals_test.append(y_test[i])

    predictions_test = np.array(predictions_test)
    actuals_test = np.array(actuals_test)
    
    # inverse transform predictions and actuals back to original scale
    predictions_original = np.zeros_like(predictions_test)
    actuals_original = np.zeros_like(actuals_test)

    for i, var in enumerate(df.columns):
        scaler = scalers[var]
        predictions_original[:, i] = scaler.inverse_transform(predictions_test[:, i].reshape(-1, 1)).flatten()
        actuals_original[:, i] = scaler.inverse_transform(actuals_test[:, i].reshape(-1, 1)).flatten()

    forecasting_variables = df.columns.tolist()
    last_actual_values = df_removed_nans_forecasting[forecasting_variables].mean().values

    # inverse differencing
    predictions_original = inverse_difference(predictions_original, last_actual_values)
    actuals_original = inverse_difference(actuals_original, last_actual_values)

    # back to dataframe format
    actuals_df = pd.DataFrame(data=actuals_original,columns=df.columns)
    predictions_df = pd.DataFrame(data=predictions_original, columns=df.columns)

    return actuals_df, filter_status_predictions(predictions_df, df.columns)


def calculate_metrics(df, actuals_original, predictions_original):
    mse = mean_squared_error(actuals_original, predictions_original)
    mae = mean_absolute_error(actuals_original, predictions_original)
    rmse = np.sqrt(mse)

    print(f"\nModel Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

    # Show actual vs predicted values
    results_df = pd.DataFrame({
        'Variable': df.columns.tolist() * actuals_original.shape[0],
        'Sample': [i for i in range(actuals_original.shape[0]) for _ in df.columns],
        'Actual': actuals_original.values.flatten(),
        'Predicted': predictions_original.values.flatten(),
        'Error': (actuals_original.values - predictions_original.values).flatten()
    })

    print(f"\nSample Results:")
    print(results_df.head(10))

def plot_results(actuals_df, predictions_df):
    plt.figure(figsize=(15*len(actuals_df.columns), 6*len(actuals_df.columns)))
    for i, column in enumerate(actuals_df.columns):
        plt.subplot(len(actuals_df.columns), 1, i+1)
        
        # Plot actual values
        plt.plot(actuals_df.index, actuals_df[column], 
                label=f'Actual {column}', color='blue', linewidth=2, alpha=0.8)
        
        # Plot predicted values
        plt.plot(predictions_df.index, predictions_df[column], 
                label=f'Predicted {column}', color='red', linewidth=2, 
                linestyle='--', alpha=0.8)
        
        plt.title(f'{column}: Actual vs Predicted')
        plt.xlabel('Time Step')
        plt.ylabel(column)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


