from get_data import get_processed_path
import warnings
import logging

import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pickle
import os


import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

import re
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import callbacks
import pickle
import numpy as np

warnings_logger = logging.getLogger('warnings')
warnings_logger.setLevel(logging.WARNING)
warning_handler = logging.FileHandler('warnings.log')
warning_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
warnings_logger.addHandler(warning_handler)

def warning_handler_func(message, category, filename, lineno, file=None, line=None):
    warnings_logger.warning(f"{category.__name__}: {message} (File: {filename}, Line: {lineno})")

warnings.showwarning = warning_handler_func

# 0. params for model training

batch_size = 32
epochs = 50

layer_one_units = 64,
layer_two_units = 128,
dense_units = 64,
activation='relu',
dropout_rate=0.2

threshold = 1000

model_description = f"Conv1D_{layer_one_units}_{layer_two_units}_Dense{dense_units}_Act{activation}_Dropout{dropout_rate}_Batch{batch_size}_Epochs{epochs}"
results_file_name = "SW-SUPV-243-classification_results_001"


# 1. encode column names

def extract_port_numbers(string_input):
    """Extract all numbers from interface string, handling multi-digit ports"""
    numbers = re.findall(r'\d+', string_input)
    return ''.join(numbers) if numbers else None

def encode_column_names(df):
    encoded_columns = {}
    unique_ports = []

    for column in df.columns:
        if "interface" in column.lower() and "/" in column:
            port_strings = column.split("/")
            port_numbers = []
            for port_string in port_strings:
                #num = extract_int(port_string)
                num = extract_port_numbers(port_string)
                if num is not None:
                    port_numbers.append(str(num))
            encoded_columns[column] = "".join(port_numbers)
        elif ": operational status" in column.lower() and "/" in column:
            encoded_columns[column] = column

    for idx, name in encoded_columns.items():
        if ": operational status" in idx.lower():
            map_name = f"Status {name}"
            encoded_columns[idx] = map_name
        elif "bits sent" in idx.lower():
            map_name = f"Bits Sent {name}"
            encoded_columns[idx] = map_name
        elif "bits received" in idx.lower():
            map_name = f"Bits Received {name}"
            encoded_columns[idx] = map_name

        if int(name) not in unique_ports:
            unique_ports.append(int(name))
    
    return encoded_columns, sorted(unique_ports)

# map ports to start from 1
def map_ports_to_start_from_one(unique_ports):
    mapped_ports = {}
    i = 1
    for port in unique_ports:
        mapped_ports[port] = i
        i+=1
    return mapped_ports

# 2. encode labels

def encode_labels(
    df,
    threshold,
    unique_ports,
    start_bit=10,
    epsilon=1e-6
):
    """
    (docstring omitted for brevity — see previous assistant message)
    """
    import pandas as pd
    import numpy as np

    unique_ports = list(unique_ports)
    df_encoded = df.copy()

    # Precompute ratios per port and record which ports are valid
    traffic_ratio_dict = {}
    valid_ports = []
    for port in unique_ports:
        sent_col = f"Bits Sent {port}"
        recv_col = f"Bits Received {port}"
        if sent_col in df.columns and recv_col in df.columns:
            sent = pd.to_numeric(df[sent_col], errors='coerce').fillna(0).astype(float)
            recv = pd.to_numeric(df[recv_col], errors='coerce').fillna(0).astype(float)
            ratio = recv / (sent + float(epsilon))
            traffic_ratio_dict[port] = ratio
            valid_ports.append(port)
        else:
            traffic_ratio_dict[port] = None
            print(f"Warning: Columns for port {port} not found in dataframe")

    contributions = []
    for i, port in enumerate(unique_ports):
        ratio = traffic_ratio_dict.get(port)
        bit_pos = start_bit + i
        if ratio is None:
            contributions.append(pd.Series(0, index=df.index, dtype=object))
            continue
        mask = (ratio > threshold)
        contribution = mask.astype(object) * (1 << bit_pos)
        contributions.append(contribution)
        print(f"Port {port} (bit {bit_pos}) - Ratio > {threshold}: {mask.sum()}")

    if contributions:
        total = pd.Series(0, index=df.index, dtype=object)
        for c in contributions:
            total = total + c
        df_encoded['Label'] = total
    else:
        df_encoded['Label'] = 0

    return df_encoded


# 3. label decoder

def decode_label(label, unique_ports, start_bit=10):
    ports = list(unique_ports)
    bit_positions = [start_bit + i for i in range(len(ports))]

    def decode_single(val):
        return {port: bool((int(val) >> bit) & 1) for port, bit in zip(ports, bit_positions)}

    if hasattr(label, "index"):  # pandas Series
        decoded = [decode_single(val) for val in label]
        return pd.DataFrame(decoded, index=label.index)
    else:
        return decode_single(label)
    
# 4. Balance Classes by Downsampling Majority Classes
def merge_small_classes(df, label_col='label', threshold=10, other_label='other'):
    class_counts = df[label_col].value_counts()
    small_classes = class_counts[class_counts < threshold].index
    df_merged = df.copy()
    df_merged[label_col] = df_merged[label_col].apply(lambda x: other_label if x in small_classes else x)
    return df_merged

# 5. Merge Small Classes
def balance_classes(df, label_col='label', random_state=42):
    class_counts = df[label_col].value_counts()
    min_count = class_counts.min()

    balanced_frames = []
    for cls in class_counts.index:
        cls_df = df[df[label_col] == cls]
        balanced_cls_df = resample(cls_df, 
                                   replace=False, 
                                   n_samples=min_count, 
                                   random_state=random_state)
        balanced_frames.append(balanced_cls_df)
    
    # Concatenate and shuffle
    balanced_df = pd.concat(balanced_frames).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return balanced_df

# 6. Prepare Data for Training

def prepare_classification_data(df_balanced_labeled):
    # simple approach of labeling for classification model
    df_classification_input = df_balanced_labeled.copy()
    unique_labels = sorted(df_classification_input['Label'].unique())
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    df_classification_input['Label'] = df_classification_input['Label'].map(label_to_index)
    num_classes = len(df_classification_input['Label'].unique())

    features_classification = df_classification_input[df_classification_input.columns[0:-1]].values
    labels_classification = df_classification_input[df_classification_input.columns[-1]].values
    # RobustScaler is better for network data (handles outliers)
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features_classification)

    # save the scaler for later use

    with open('C:\\ThesisWork\\offical_approach\\mth_project\\mth_project\\industrial_network_analysis\\classification_model\\scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return features_scaled, labels_classification, label_to_index, unique_labels, num_classes

def create_windows(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size + 1):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size-1])  # label from last time step in window
    return np.array(Xs), np.array(ys)


def data_split(features_scaled, labels_classification, window_size):
    X_seq, y_seq = create_windows(features_scaled, labels_classification, window_size)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq,
        test_size=0.2,
        stratify=y_seq,   # ensures same label distribution in both
        random_state=42
    )
    return X_train, X_test, y_train, y_test

# 7. Build and Train the Model

# BASIC MODEL:

def build_classification_model(X_seq,
                                num_classes, 
                                window_size,
                                layer_one_units = 64,
                                layer_two_units = 128,
                                dense_units = 64,
                                activation='relu',
                                dropout_rate=0.3
                            ):
    input_shape=(window_size, X_seq.shape[2])
    model = models.Sequential([
        layers.Conv1D(layer_one_units, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv1D(layer_two_units, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(dense_units, activation=activation),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model

def train_classification_model(model, X_train, y_train, X_test, y_test, epochs = 10, batch_size=16):
    # callbacks for better training

    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'C:\\ThesisWork\\offical_approach\\mth_project\\mth_project\\industrial_network_analysis\\classification_model\\best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # updated training call
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,  # More epochs with early stopping
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )

    y_pred = model.predict(X_test).argmax(axis=1)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    return history, y_pred, test_loss, test_accuracy

if __name__ == "__main__":
    ### 0. Get the same Data as in Forecasting part
    print("STEP 0/7: Loading data...")

    processed_forecasting_path, processed_statuses_path = get_processed_path()

    df_removed_nans_forecasting = pd.read_csv(processed_forecasting_path, index_col=0, parse_dates=True)
    df_removed_nans_classification = pd.read_csv(processed_statuses_path, index_col=0, parse_dates=True)
    # to do in future: train model on input data from different dates
    # merge data

    # df = pd.merge(df_removed_nans_forecasting, df_removed_nans_classification, on=['timestamp'])

    # only use forecasting data for classification

    df = df_removed_nans_forecasting.copy()

    print(f"    Merged DataFrame shape: {df.shape}")

    ### ad 1. Encode Column Names
    print("STEP 1/7: Encoding column names...")

    encoded_columns, unique_ports = encode_column_names(df)

    # update DataFrame with encoded column names
    df.rename(columns=encoded_columns, inplace=True)

    ### ad 2. Encode Labels
    print("STEP 2/7: Encoding labels...")

    storm_threshold = 100
    
    mapped_ports = map_ports_to_start_from_one(unique_ports)
    df_labeled = encode_labels(df, threshold = storm_threshold, unique_ports=unique_ports)

    balancing = False

    ### ad 3. Balance Classes by Downsampling Majority Classes
    print("STEP 3/7: Merging small classes...")
    df_merged_labeled = merge_small_classes(df_labeled, label_col='Label', threshold=threshold, other_label='0000')
    print(f"    results for threshold: {threshold} = {df_merged_labeled['Label'].value_counts()}")

    # unify datatype for label column
    df_merged_labeled['Label'] = df_merged_labeled['Label'].astype(int)

    if balancing:
        ### ad 4. Merge Small Classes
        print("STEP 4/7: Balancing classes...")
        df_balanced_labeled = balance_classes(df_merged_labeled, label_col='Label', random_state=42)
        print(f"    {df_balanced_labeled['Label'].value_counts()}")
    else:
        print("Skipping step 4: No balancing applied.")

    ### ad 5. Decode Labels - AFTER merging to include all final labels
    print("STEP 5/7: Decoding final labels...")
    label_to_name = {}
    #final_labels = df_balanced_labeled['Label'].unique()
    final_labels = df_labeled['Label'].unique()

    label_to_name = {}

    for label in df_labeled['Label'].unique()[:5]:
        decoded = decode_label(label, unique_ports)
        decoded = {k: v for k, v in decoded.items() if v}  
        label_to_name[label] = decoded
        print(f"Label: {label} -> Decoded: {decoded}")

    ### ad 6. Prepare Data for Training
    print("STEP 6/7: Preparing data for training...")
    df_for_training = df_balanced_labeled if balancing else df_merged_labeled
    features_scaled, labels_classification, label_to_index, unique_labels, num_classes = prepare_classification_data(df_for_training)
    window_size = 6
    X_train, X_test, y_train, y_test = data_split(features_scaled, labels_classification, window_size)

    ### ad 7. Build and Train the Model
    print("STEP 7/7: Building and training the model...")
    model = build_classification_model(X_train, num_classes, window_size)

    history, y_pred, test_loss, test_accuracy = train_classification_model(model, X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)

    print(f"    Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    ### 8. Save Model and Encoders
    classification_path = "C:\\ThesisWork\\offical_approach\\mth_project\\mth_project\\industrial_network_analysis\\classification_model"
    model.save(f'{classification_path}\\final_model.h5')
    encoder_data = {}
    encoder_data['label_to_index'] = label_to_index
    encoder_data['index_to_label'] = {v: k for k, v in label_to_index.items()}
    encoder_data['label_to_name'] = label_to_name
    with open(f'{classification_path}\\encoders.pkl', 'wb') as f:
        pickle.dump(encoder_data, f)

    print(f"Model and encoders saved under {classification_path}")

    description = f"Model Description: {model_description}, \n \
    Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f} \n \
        Threshold for merging small classes: {threshold}, \n \
        Amount of classes: {num_classes}"
    with open(f"{classification_path}\\{results_file_name}.txt", "w") as f:
        f.write(f"Model Description: {description}\n")
        