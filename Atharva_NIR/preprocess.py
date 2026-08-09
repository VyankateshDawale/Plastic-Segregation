import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.signal import savgol_filter
import joblib
import os
import json

class SNVNormalizer:
    """Standard Normal Variate (SNV) Normalizer for spectral data."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Apply SNV row-wise: (x - mean) / std
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        # Avoid division by zero
        std[std == 0] = 1e-8
        return (X - mean) / std
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

def preprocess_data():
    print("Starting data preprocessing...")
    
    # 1. Load dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    csv_path = os.path.join(project_root, 'FTIR-PLASTIC-c4', 'FTIR_PLASTIC_c4.csv')
    if not os.path.exists(csv_path):
        # Fallback to local
        csv_path = 'FTIR-PLASTIC-c4/FTIR_PLASTIC_c4.csv'
    df = pd.read_csv(csv_path)
    
    # 2. Extract columns
    y_cols = ['Data(y)'] + [f'Data(y).{i}' for i in range(1, 3736)]
    X = df[y_cols].values
    y = df['Polymer'].values
    
    # 3. Handle duplicates
    # We find duplicate rows based on spectral features
    _, unique_indices = np.unique(X, axis=0, return_index=True)
    # Sort indices to maintain original order
    unique_indices = sorted(unique_indices)
    
    X_unique = X[unique_indices]
    y_unique = y[unique_indices]
    print(f"Removed duplicates: original shape {X.shape}, unique shape {X_unique.shape}")
    
    # 4. Split train/validation/test (stratified)
    # 70% train, 15% val, 15% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_unique, y_unique, test_size=0.15, stratify=y_unique, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42 # 0.1765 of 85% is 15% of total
    )
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    
    # 5. Preprocessing pipeline fit and transform
    # Step A: Savitzky-Golay smoothing (window_length=15, polyorder=2, deriv=0)
    # We apply this first to smooth the raw spectra
    print("Applying Savitzky-Golay smoothing...")
    X_train_sg = savgol_filter(X_train, window_length=15, polyorder=2, deriv=0, axis=1)
    X_val_sg = savgol_filter(X_val, window_length=15, polyorder=2, deriv=0, axis=1)
    X_test_sg = savgol_filter(X_test, window_length=15, polyorder=2, deriv=0, axis=1)
    
    # Step B: SNV Normalization (row-wise)
    print("Applying Standard Normal Variate (SNV) normalization...")
    snv = SNVNormalizer()
    X_train_snv = snv.fit_transform(X_train_sg)
    X_val_snv = snv.transform(X_val_sg)
    X_test_snv = snv.transform(X_test_sg)
    
    # Step C: Standard Scaler (column-wise)
    print("Applying column-wise StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_snv)
    X_val_scaled = scaler.transform(X_val_snv)
    X_test_scaled = scaler.transform(X_test_snv)
    
    # 6. Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)
    
    # 7. Save pipeline components and datasets
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    models_dir = os.path.join(project_root, 'models')
    results_dir = os.path.join(project_root, 'results')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save transformers
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.pkl'))
    print(f"Saved scaler.pkl and label_encoder.pkl to {models_dir}")
    
    # Save the split datasets as npz for easy loading
    np.savez(os.path.join(results_dir, 'dataset_split.npz'), 
             X_train=X_train_scaled, y_train=y_train_enc,
             X_val=X_val_scaled, y_val=y_val_enc,
             X_test=X_test_scaled, y_test=y_test_enc,
             classes=label_encoder.classes_)
    
    # Save label mapping for inference reference
    label_mapping = {int(i): str(c) for i, c in enumerate(label_encoder.classes_)}
    with open(os.path.join(results_dir, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f, indent=4)
        
    print("Preprocessing pipeline finished successfully!")

if __name__ == "__main__":
    preprocess_data()
