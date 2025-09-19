from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import time
import sys
import os
import pickle
from get_data import get_processed_path

from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras

# model parameters that can be adjusted or changed for testing purpose
epochs = 50
batch_size = 32
validation_split = 0.2
verbose = 1
context_length = 60    # FIXED!
first_layer_units = 128
second_layer_units = 64
dense_units = 256
activation = 'relu'
dropout_rate = 0.2

model_description = f"Epochs: {epochs}, Batch Size: {batch_size}, Validation Split: {validation_split}, Context Length: {context_length}, First Layer Units: {first_layer_units}, Second Layer Units: {second_layer_units}, Dense Units: {dense_units}, Activation: {activation}, Dropout Rate: {dropout_rate}"
results_file_name = "initial_model_results"

def create_online_multivariate_model(df,
                                     context_length=context_length,
                                     first_layer_units=first_layer_units,
                                     second_layer_units=second_layer_units,
                                     dense_units=dense_units,
                                     activation=activation,
                                     dropout_rate=dropout_rate):
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


def split_data_for_initial_model(df, context_length=context_length):
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

def train_initial_model(model, X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split = validation_split, verbose = verbose):

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose
    )

    # early stopping?


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
    percentage_error = np.mean(np.abs((actuals_original - predictions_original) / actuals_original)) * 100

    print(f"\nModel Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Percentage Error: {percentage_error:.6f}")

    # Show actual vs predicted values
    results_df = pd.DataFrame({
        'Variable': df.columns.tolist() * actuals_original.shape[0],
        'Sample': [i for i in range(actuals_original.shape[0]) for _ in df.columns],
        'Actual': actuals_original.values.flatten(),
        'Predicted': predictions_original.values.flatten(),
        'Error': (actuals_original.values - predictions_original.values).flatten()
    })

    print(f"\nSample Results:")
    print(results_df)
    return results_df, mse, mae, rmse, percentage_error

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

# save online data, needed for main program and online forecasting
def save_online_data(initial_model_path, df_online, scalers_train, context_length, df_removed_nans_forecasting, df_removed_nans_classification, variables):
    """
    Save online data in their original formats to preserve structure and metadata.
    
    Args:
        initial_model_path: Path to save the data
        df_online: pandas DataFrame with online forecasting data
        scalers_train: dict of sklearn scalers
        context_length: int
        df_removed_nans_forecasting: pandas DataFrame with forecasting data
        df_removed_nans_classification: pandas DataFrame with classification data
        variables: pandas Index or list of column names
    """
    
    # Create directory if it doesn't exist
    os.makedirs(initial_model_path, exist_ok=True)
    
    # Save DataFrames as CSV or Parquet to preserve structure
    df_online.to_csv(f"{initial_model_path}\\df_online.csv", index=True)
    df_removed_nans_forecasting.to_csv(f"{initial_model_path}\\df_removed_nans_forecasting.csv", index=True)
    df_removed_nans_classification.to_csv(f"{initial_model_path}\\df_removed_nans_classification.csv", index=True)
    
    # Save scalers dictionary using pickle (preserves sklearn objects)
    import pickle
    with open(f"{initial_model_path}\\scalers_train.pkl", 'wb') as f:
        pickle.dump(scalers_train, f)
    
    # Save simple values as numpy (these are fine as numpy)
    np.save(f"{initial_model_path}\\context_length.npy", context_length)
    
    # Save variables as list (preserves column names)
    variables_list = list(variables) if hasattr(variables, 'tolist') else list(variables)
    with open(f"{initial_model_path}\\variables.txt", 'w') as f:
        for var in variables_list:
            f.write(f"{var}\n")
    
    print(f"✓ Online data saved in original formats to: {initial_model_path}")
    print(f"  - DataFrames saved as CSV files")
    print(f"  - Scalers saved as pickle file")
    print(f"  - Variables saved as text file")
    print(f"  - Context length saved as numpy file")

def get_initial_model(initial_model_path=None):
    """
    Load the trained initial model from the specified path.
    
    Args:
        initial_model_path (str, optional): Path to the model directory. 
                                           If None, uses the default path.
    
    Returns:
        keras.Model: The loaded trained model
    """
    if initial_model_path is None:
        initial_model_path = "C:\\ThesisWork\\offical_approach\\mth_project\\mth_project\\industrial_network_analysis\\forecasting_model"
    
    model_file_path = f"{initial_model_path}\\initial_model.h5"
    
    # Check if model file exists
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found at: {model_file_path}")
    
    try:
        # Try loading with custom objects to handle version compatibility
        custom_objects = {
            'mse': keras.losses.MeanSquaredError(),
            'mae': keras.metrics.MeanAbsoluteError(),
            'mean_squared_error': keras.losses.MeanSquaredError(),
            'mean_absolute_error': keras.metrics.MeanAbsoluteError(),
        }
        
        try:
            # First try loading with custom objects
            model = keras.models.load_model(model_file_path, custom_objects=custom_objects)
        except Exception as e1:
            print(f"First attempt failed: {e1}")
            try:
                # Second attempt: Load without compilation
                model = keras.models.load_model(model_file_path, compile=False)
                # Recompile the model with current Keras version
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss=keras.losses.MeanSquaredError(),
                    metrics=[keras.metrics.MeanAbsoluteError()]
                )
                print("Model loaded without compilation and recompiled successfully")
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                raise e2
        
        # Basic model validation
        if model is None:
            raise ValueError("Loaded model is None")
        
        # Check if model has the expected structure
        if not hasattr(model, 'layers') or len(model.layers) == 0:
            raise ValueError("Loaded model appears to be invalid (no layers)")
        
        print(f"✓ Model loaded successfully from: {model_file_path}")
        print(f"  - Model type: {type(model)}")
        print(f"  - Number of layers: {len(model.layers)}")
        print(f"  - Input shape: {model.input_shape if hasattr(model, 'input_shape') else 'Unknown'}")
        print(f"  - Output shape: {model.output_shape if hasattr(model, 'output_shape') else 'Unknown'}")
        
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_file_path}: {str(e)}")

def get_online_data(initial_model_path):
    """
    Load online data in their original formats.
    
    Args:
        initial_model_path: Path to load the data from
        
    Returns:
        tuple: (df_online, scalers_train, context_length, df_removed_nans_forecasting, 
                df_removed_nans_classification, variables)
    """
    
    # Check if files exist and determine which format to use
    csv_format = os.path.exists(f"{initial_model_path}\\df_online.csv")
    npy_format = os.path.exists(f"{initial_model_path}\\df_online.npy")
    
    if csv_format:
        # Load DataFrames from CSV (preserves original structure)
        df_online = pd.read_csv(f"{initial_model_path}\\df_online.csv", index_col=0)
        df_removed_nans_forecasting = pd.read_csv(f"{initial_model_path}\\df_removed_nans_forecasting.csv", index_col=0)
        df_removed_nans_classification = pd.read_csv(f"{initial_model_path}\\df_removed_nans_classification.csv", index_col=0)
        
        # Load scalers from pickle (preserves sklearn objects)
        import pickle
        with open(f"{initial_model_path}\\scalers_train.pkl", 'rb') as f:
            scalers_train = pickle.load(f)
        
        # Load context length from numpy
        context_length = np.load(f"{initial_model_path}\\context_length.npy", allow_pickle=True).item()
        
        # Load variables from text file
        with open(f"{initial_model_path}\\variables.txt", 'r') as f:
            variables = [line.strip() for line in f.readlines() if line.strip()]
            
        print(f"✓ Online data loaded from original formats at: {initial_model_path}")
        print(f"  - DataFrames loaded from CSV files")
        print(f"  - Scalers loaded from pickle file") 
        print(f"  - Variables loaded from text file")
        
    elif npy_format:
        # Fallback: Load from old numpy format (for backward compatibility)
        print("⚠ Loading from legacy numpy format. Consider regenerating data for better compatibility.")
        
        df_online_data = np.load(f"{initial_model_path}\\df_online.npy", allow_pickle=True)
        df_forecasting_data = np.load(f"{initial_model_path}\\df_removed_nans_forecasting.npy", allow_pickle=True)
        df_classification_data = np.load(f"{initial_model_path}\\df_removed_nans_classification.npy", allow_pickle=True)
        
        # Load dictionary and scalar data with .item()
        scalers_train = np.load(f"{initial_model_path}\\scalers_train.npy", allow_pickle=True).item()
        context_length = np.load(f"{initial_model_path}\\context_length.npy", allow_pickle=True).item()
        
        # Handle DataFrame loading - check if they were saved as object arrays (containing DataFrames) or regular arrays
        if df_online_data.dtype == 'object' and df_online_data.ndim == 0:
            df_online = df_online_data.item()
        else:
            df_online = df_online_data
        
        if df_forecasting_data.dtype == 'object' and df_forecasting_data.ndim == 0:
            df_removed_nans_forecasting = df_forecasting_data.item()
        else:
            df_removed_nans_forecasting = df_forecasting_data
            
        if df_classification_data.dtype == 'object' and df_classification_data.ndim == 0:
            df_removed_nans_classification = df_classification_data.item()
        else:
            df_removed_nans_classification = df_classification_data

        variables = np.load(f"{initial_model_path}\\variables.npy", allow_pickle=True).tolist()
        
    else:
        raise FileNotFoundError(f"No data files found at: {initial_model_path}")

    return df_online, scalers_train, context_length, df_removed_nans_forecasting, df_removed_nans_classification, variables


if __name__ == "__main__":

    ### create initial model

    ### READ DATA
    processed_forecasting_path, processed_statuses_path = get_processed_path()

    df_removed_nans_forecasting = pd.read_csv(processed_forecasting_path)
    df_removed_nans_classification = pd.read_csv(processed_statuses_path)

    df_removed_nans_forecasting = df_removed_nans_forecasting.select_dtypes(include=[np.number])
    df_removed_nans_classification = df_removed_nans_classification.select_dtypes(include=[np.number])

    # differenciate data for forecasting

    df_differenced = df_removed_nans_forecasting.diff().dropna()

    # Split data for initial training and online forecasting

    initial_idx = 24 * 60 # first 24 hours for initial training
    df_initial = df_differenced.iloc[:initial_idx].copy()
    df_online = df_differenced.iloc[initial_idx:].copy()

    variables = df_initial.columns

    model = create_online_multivariate_model(
        df=df_initial,
        context_length=context_length,
        first_layer_units=first_layer_units,
        second_layer_units=second_layer_units,
        dense_units=dense_units,
        activation=activation,
        dropout_rate=dropout_rate
    )

    # split data for initial model training

    split_ratio = 0.8

    df_train = df_initial.iloc[:int(split_ratio * len(df_initial))]
    df_test = df_initial.iloc[int(split_ratio * len(df_initial)):]

    X_train, y_train, scalers_train = split_data_for_initial_model(df_train)
    X_test, y_test, scalers_test = split_data_for_initial_model(df_test)


    print("======================================================")
    print("Training initial model...")
    print("======================================================")

    history, initial_model = train_initial_model(model, X_train, y_train, epochs = epochs)

    # save initial model
    initial_model_path = "C:\\ThesisWork\\offical_approach\\mth_project\\mth_project\\industrial_network_analysis\\forecasting_model"
    initial_model.save(f"{initial_model_path}\\initial_model.h5")

    df_actuals, df_predictions = test_initial_model(model, df_initial, X_test, y_test, scalers_test, df_removed_nans_forecasting)
    results_df, mse, mae, rmse, percentage_error = calculate_metrics(df_initial, df_actuals, df_predictions)
    save_online_data(f"{initial_model_path}\\online_data", df_online, scalers_train, context_length, df_removed_nans_forecasting, df_removed_nans_classification, variables)
    #plot_results(df_actuals, df_predictions)

    description = f"Initial Model Training Description:\n{model_description}\n Results:\n MSE: {mse:.6f}\n MAE: {mae:.6f}\n RMSE: {rmse:.6f}\n Percentage Error: {percentage_error:.6f}\n"
    with open(f"{initial_model_path}\\{results_file_name}.txt", "w") as f:
        f.write(description)