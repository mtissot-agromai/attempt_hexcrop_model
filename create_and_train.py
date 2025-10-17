import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.interpolate import interp1d
import glob
import os
import random
from typing import List, Dict, Union

# --- Configuration Parameters ---
NUM_STEPS = 40  # The fixed sequence length for the DL model
NUM_CLASSES = 4 # THIS CHANGED: Set to 4 for bean, soybean, rice, maize
COLUMNS_TO_KEEP = ['cycle_length',
                   'B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12',
                   'NDVI','NDMI','NDWI',
                   'culture']

FILE_PATH_GLOB = 'data_files/*.csv'

CROP_MAPPING = {
    'bean': 0, 
    'soybean': 1, 
    'rice': 2, 
    'maize': 3
}
# --------------------------------

def standardize_time_series_for_dl(
    df: pd.DataFrame, 
    spectral_columns: List[str],
    fixed_length: int = NUM_STEPS, 
    date_column: str = 'date'
) -> np.ndarray:
    
    df[date_column] = pd.to_datetime(df[date_column])
    
    df_clean = df.dropna(subset=spectral_columns + [date_column]).sort_values(date_column).copy()
    
    if df_clean.empty or len(df_clean) < 2:
        return np.full((fixed_length, len(spectral_columns)), 0.0) # Fill with zero if insufficient data

    sowing_date = df_clean[date_column].min()
    df_clean['DSS'] = (df_clean[date_column] - sowing_date).dt.days
    
    # Define the range of the new uniform time steps (0 to max DSS)
    max_dss = df_clean['DSS'].max()
    new_dss_steps = np.linspace(0, max_dss, fixed_length)
    
    standardized_data = np.zeros((fixed_length, len(spectral_columns)))
    
    # 2. Resample/Interpolate each spectral column
    for i, col in enumerate(spectral_columns):
        t_original = df_clean['DSS'].values
        y_original = df_clean[col].values
        
        # Use linear interpolation
        f = interp1d(t_original, y_original, kind='linear', fill_value='extrapolate')
        
        y_resampled = f(new_dss_steps)
        
        # Min-Max normalization for DL stability across the 0-1 range
        min_val = y_resampled.min()
        max_val = y_resampled.max()
        if max_val != min_val:
            y_resampled = (y_resampled - min_val) / (max_val - min_val)
        
        standardized_data[:, i] = y_resampled
        
    return standardized_data

def define_and_train_model(X: np.ndarray, y: np.ndarray, num_features: int, num_classes: int) -> keras.Model:
    
    # Convert labels to one-hot encoding
    y_encoded = keras.utils.to_categorical(y, num_classes=num_classes)
    
    # Define a simple 1D CNN model structure
    model = keras.Sequential([
        keras.layers.Input(shape=(NUM_STEPS, num_features)),
        # Feature extraction via convolution across the time dimension
        keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        keras.layers.MaxPool1D(pool_size=2),
        keras.layers.Dropout(0.3),
        
        keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        keras.layers.MaxPool1D(pool_size=2),
        keras.layers.Dropout(0.3),
        
        # Collapse the temporal dimension
        keras.layers.GlobalAveragePooling1D(),
        
        # Classification head
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Simple train/test split
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y_encoded[:split_index], y_encoded[split_index:]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=10, # Low epochs for example, increase this in production
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0 # Set to 1 or 2 to see training progress
    )
    
    print(f"Model trained. Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model

# Placeholder for user's file creation/loop logic
def create_dataset_and_train():
    
    os.makedirs('data_files', exist_ok=True)
    # THIS CHANGED: Use string labels from the keys of the mapping
    crop_labels = list(CROP_MAPPING.keys()) 
    # for i in range(20):
    #     N_obs = random.randint(25, 60) # Variable length
    #     data = {col: np.random.rand(N_obs) for col in COLUMNS_TO_KEEP[:-1]}
    #     data['date'] = pd.date_range(start='2023-04-01', periods=N_obs, freq='7D') # Irregular time steps
    #     # THIS CHANGED: Assign a random string label
    #     data['culture'] = random.choice(crop_labels) 
        
    #     df = pd.DataFrame(data)
    #     df.to_csv(f'data_files/field_{i}.csv', index=False)
    # print("MOCK DATA GENERATION COMPLETE (20 files with variable lengths and string labels).")
    # ----------------------------------------------------
    
    # --- 1. & 2. Main Data Loading and Standardization Loop ---
    
    X_list = []
    y_list = []
    
    spectral_cols = COLUMNS_TO_KEEP[:-1] # All columns except 'label'
    
    # THIS CHANGED: Loop through files using glob
    for filepath in glob.glob(FILE_PATH_GLOB):
        try:
            df = pd.read_csv(filepath)
            
            # THIS CHANGED: Check if the culture column is present and extract it
            if 'culture' not in df.columns or df['culture'].isnull().all():
                print(f"Skipping {filepath}: Missing culture.")
                continue

            # Standardize the time series to a fixed length (e.g., 40 steps)
            X_standardized = standardize_time_series_for_dl(
                df, 
                spectral_cols,
                fixed_length=NUM_STEPS
            )
            
            # Use the first valid culture in the file (assuming culture is constant per file/field)
            y_str_culture = df['culture'].dropna().iloc[0]
            
            # THIS CHANGED: Convert string culture to integer using the CROP_MAPPING
            y_culture = CROP_MAPPING.get(y_str_culture, -1) 
            if y_culture == -1:
                print(f"Skipping {filepath}: Unknown culture '{y_str_culture}'.")
                continue

            X_list.append(X_standardized)
            y_list.append(y_culture)
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
    # --- 3. Create the Final Tensors ---
    
    if not X_list:
        print("No valid data files found or processed. Aborting.")
        return

    # THIS CHANGED: Convert list of (T x D) arrays into a single (N x T x D) tensor
    X_tensor = np.stack(X_list).astype(np.float32)
    y_vector = np.array(y_list).astype(np.int32)
    
    print(f"\nFinal Feature Tensor Shape (N x T x D): {X_tensor.shape}")
    print(f"Final Label Vector Shape (N): {y_vector.shape}")
    
    # --- 4. & 5. Define and Train the 1D CNN Model ---
    
    num_features = X_tensor.shape[2]
    model = define_and_train_model(X_tensor, y_vector, num_features, NUM_CLASSES)
    
    # --- 6. Export to ONNX Format ---
    
    try:
        import tf2onnx
        import onnx
        
        model_filepath = 'crop_classifier.onnx'

        # Define the input signature required by ONNX
        spec = (tf.TensorSpec((None, NUM_STEPS, num_features), tf.float32, name="input"),)
        
        # Convert and save
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        onnx.save(model_proto, model_filepath)
        
        print(f"\nModel successfully exported to {model_filepath}")
        
    except ImportError:
        print("\nSkipping ONNX export: 'tf2onnx' or 'onnx' library not found.")
        print("Please run: pip install tf2onnx onnx")

if __name__ == '__main__':
    create_dataset_and_train()