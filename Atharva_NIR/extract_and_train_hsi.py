import os
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
import scipy.io
import joblib
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def extract_hsi_data():
    print("==================================================")
    print("STEP 1: Extracting HSI subimages from RAR archive...")
    print("==================================================")
    
    # We extract Train, Val, and Test folders from the RAR file
    # This reads the 42.6 GB archive only once, using ~9.3 GB of disk space.
    cmd = [
        "tar", "-xf", "ADVANCED_PLASTIC_FUSE.rar",
        "ADVANCED_PLASTIC_FUSE/Train",
        "ADVANCED_PLASTIC_FUSE/Val",
        "ADVANCED_PLASTIC_FUSE/Test"
    ]
    print("Running extraction command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Extraction failed:", result.stderr)
        raise RuntimeError("Failed to extract RAR file.")
    print("Extraction completed successfully!")

def process_hsi_files():
    print("\n==================================================")
    print("STEP 2: Processing MAT files and extracting mean spectra...")
    print("==================================================")
    
    data_records = []
    
    # We search in Train, Val, and Test folders
    for split in ["Train", "Val", "Test"]:
        folder_path = os.path.join("ADVANCED_PLASTIC_FUSE", split, "*.mat")
        files = glob.glob(folder_path)
        print(f"Processing {len(files)} files in split '{split}'...")
        
        for idx, fpath in enumerate(files):
            # Filename format: prefix_label_... e.g. 1_HDPE_...
            fname = os.path.basename(fpath)
            parts = fname.split('_')
            if len(parts) < 2:
                continue
                
            prefix = parts[0] + "_" + parts[1]
            # Map prefix to standardized label
            label_map = {
                "1_HDPE": "HDPE",
                "2_PP": "PP",
                "3_ORGANIC": "ORGANIC",
                "4_LDPE": "LDPE",
                "5_PET": "PET",
                "6_OTHER": "OTHER",
                "7_PS": "PS",
                "7_Ps": "PS"
            }
            label = label_map.get(prefix, "OTHER")
            
            try:
                # Load MATLAB file
                mat = scipy.io.loadmat(fpath)
                subimage = mat['subimage'] # Shape: (64, 64, 232)
                
                # Compute mean spectrum over spatial dimensions (64, 64) -> (232,)
                mean_spectrum = subimage.mean(axis=(0, 1))
                
                data_records.append({
                    "filename": fname,
                    "split": split,
                    "label": label,
                    "features": mean_spectrum
                })
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                
            # Log progress
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(files)} files...")
                
    # Clean up extracted folder immediately to free disk space
    print("Cleaning up extracted directories...")
    shutil.rmtree("ADVANCED_PLASTIC_FUSE", ignore_errors=True)
    print("Disk space cleaned up successfully!")
    
    return data_records

def train_and_evaluate_hsi(data_records):
    print("\n==================================================")
    print("STEP 3: Preparing dataset and training models...")
    print("==================================================")
    
    # Separate splits
    X_train_raw = []
    y_train_raw = []
    X_val_raw = []
    y_val_raw = []
    X_test_raw = []
    y_test_raw = []
    
    for r in data_records:
        if r["split"] == "Train":
            X_train_raw.append(r["features"])
            y_train_raw.append(r["label"])
        elif r["split"] == "Val":
            X_val_raw.append(r["features"])
            y_val_raw.append(r["label"])
        elif r["split"] == "Test":
            X_test_raw.append(r["features"])
            y_test_raw.append(r["label"])
            
    X_train = np.array(X_train_raw)
    y_train = np.array(y_train_raw)
    X_val = np.array(X_val_raw)
    y_val = np.array(y_val_raw)
    X_test = np.array(X_test_raw)
    y_test = np.array(y_test_raw)
    
    print(f"Shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)
    
    # Save preprocessing scale and encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler_nir_hsi.pkl')
    joblib.dump(label_encoder, 'models/label_encoder_nir_hsi.pkl')
    print("Saved HSI scaler and label encoder to models/")
    
    # Train classical models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='rbf', C=10.0, probability=True, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    
    results = {}
    best_f1 = 0
    best_model = None
    best_model_name = ""
    
    for name, model in models.items():
        print(f"Training HSI model {name}...")
        model.fit(X_train_scaled, y_train_enc)
        
        y_test_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test_enc, y_test_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test_enc, y_test_pred, average='weighted')
        
        results[name] = {
            "test_accuracy": float(acc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_f1": float(f1)
        }
        print(f"  {name} - Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name
            
    # Save the best model as the final NIR model
    joblib.dump(best_model, 'models/nir_hsi_model.pkl')
    print(f"\nBest model selected: {best_model_name} (F1-score: {best_f1:.4f})")
    print("Saved HSI model to models/nir_hsi_model.pkl")
    
    # Save results to json
    os.makedirs('results', exist_ok=True)
    with open('results/nir_hsi_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved HSI training results to results/nir_hsi_results.json")
    print("HSI pipeline completed successfully!")

if __name__ == "__main__":
    extract_hsi_data()
    records = process_hsi_files()
    train_and_evaluate_hsi(records)
