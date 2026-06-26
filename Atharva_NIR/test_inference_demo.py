import os
import subprocess
import shutil
import numpy as np
import pandas as pd
import scipy.io
from inference.inference import DualStagePlasticInference

def extract_and_load_hsi_sample(archive_path, mat_path):
    """Helper to extract a single MAT file from the RAR archive and return its mean spectrum."""
    cmd = ["tar", "-xf", archive_path, mat_path]
    subprocess.run(cmd, check=True)
    
    try:
        mat = scipy.io.loadmat(mat_path)
        subimage = mat['subimage'] # shape (64, 64, 232)
        mean_spectrum = subimage.mean(axis=(0, 1))
    finally:
        shutil.rmtree(mat_path.split('/')[0], ignore_errors=True)
        
    return mean_spectrum

def main():
    print("==================================================")
    print("DEMO: Dual-Stage NIR to MIR Fallback Verification")
    print("==================================================")
    
    # Initialize the inference pipeline (confidence threshold = 85%, black plastic reflectance threshold = 0.08)
    pipeline = DualStagePlasticInference(models_dir='models', confidence_threshold=0.85, black_plastic_threshold=0.08)
    
    archive_path = "ADVANCED_PLASTIC_FUSE.rar"
    mir_csv_path = "FTIR-PLASTIC-c4/FTIR_PLASTIC_c4.csv"
    
    # Load clean MIR spectra from FTIR dataset
    df_mir = pd.read_csv(mir_csv_path)
    y_cols_mir = ['Data(y)'] + [f'Data(y).{i}' for i in range(1, 3736)]
    
    # --------------------------------------------------
    # CASE 1: Standard plastic scanned via NIR (Conclusive result)
    # --------------------------------------------------
    print("\n" + "="*50)
    print("CASE 1: Scanning a standard colored plastic sample (PET)")
    print("="*50)
    
    pet_mat_path = "ADVANCED_PLASTIC_FUSE/Train/5_PET_13_PET_BOTTLES_FX10_0000_1.mat"
    print(f"Extracting HSI sample: {pet_mat_path}...")
    raw_nir_pet = extract_and_load_hsi_sample(archive_path, pet_mat_path)
    
    # Run prediction
    result_pet = pipeline.classify_sample(raw_nir=raw_nir_pet)
    
    print("\n--- FINAL CLASSIFICATION REPORT ---")
    print(f"Sensor Used: {result_pet['sensor_used']}")
    print(f"Predicted Class: {result_pet['predicted_class']}")
    print(f"Confidence: {result_pet['confidence'] * 100:.2f}%")
    print(f"Fallback Triggered: {result_pet['fallback_triggered']}")
    print("Status: SUCCESS [OK]" if result_pet['predicted_class'] == 'PET' else "Status: FAILED [ERROR]")
    
    # --------------------------------------------------
    # CASE 2: Black plastic (Low NIR reflectance -> MIR Fallback)
    # --------------------------------------------------
    print("\n" + "="*50)
    print("CASE 2: Scanning a black HDPE sample (Low reflectance -> MIR Fallback)")
    print("="*50)
    
    # We simulate a black plastic scan by scaling down a normal HDPE spectrum's reflectance to ~0.02
    hdpe_mat_path = "ADVANCED_PLASTIC_FUSE/Train/1_HDPE_2_HDPE_BOTTLES_FX10_0000_1.mat"
    print(f"Extracting HSI sample: {hdpe_mat_path}...")
    raw_nir_hdpe = extract_and_load_hsi_sample(archive_path, hdpe_mat_path)
    
    # Scale down raw NIR values to simulate the severe light absorption of carbon black
    black_hdpe_nir = raw_nir_hdpe * 0.04  # Brings mean reflectance well below the 0.08 threshold
    
    # Load backup high-resolution HDPE spectrum from MIR dataset
    hdpe_row = df_mir[df_mir['Polymer'] == 'HDPE'].iloc[0]
    raw_mir_hdpe = hdpe_row[y_cols_mir].values.astype(float)
    
    # Run prediction with low-intensity NIR and clean MIR spectrum
    result_black = pipeline.classify_sample(raw_nir=black_hdpe_nir, raw_mir=raw_mir_hdpe)
    
    print("\n--- FINAL CLASSIFICATION REPORT ---")
    print(f"Sensor Used: {result_black['sensor_used']}")
    print(f"Predicted Class: {result_black['predicted_class']}")
    print(f"Confidence: {result_black['confidence'] * 100:.2f}%")
    print(f"NIR Analysis: {result_black['nir_prediction']}")
    print(f"Fallback Reason: {result_black['fallback_reason']}")
    print(f"Fallback Triggered: {result_black['fallback_triggered']}")
    print("Status: SUCCESS [OK]" if result_black['predicted_class'] == 'HDPE' else "Status: FAILED [ERROR]")
    
    # --------------------------------------------------
    # CASE 3: Non-plastic contaminant scanned (Organic -> MIR Fallback)
    # --------------------------------------------------
    print("\n" + "="*50)
    print("CASE 3: Scanning organic waste contaminant (Organic -> MIR Fallback check)")
    print("="*50)
    
    organic_mat_path = "ADVANCED_PLASTIC_FUSE/Train/3_ORGANIC_12_LDPE_PLANT_FX10_0000_1.mat"
    print(f"Extracting HSI sample: {organic_mat_path}...")
    raw_nir_org = extract_and_load_hsi_sample(archive_path, organic_mat_path)
    
    # Load backup PVC spectrum from MIR dataset
    pvc_row = df_mir[df_mir['Polymer'] == 'PVC'].iloc[0]
    raw_mir_pvc = pvc_row[y_cols_mir].values.astype(float)
    
    # Run prediction
    result_fallback = pipeline.classify_sample(raw_nir=raw_nir_org, raw_mir=raw_mir_pvc)
    
    print("\n--- FINAL CLASSIFICATION REPORT ---")
    print(f"Sensor Used: {result_fallback['sensor_used']}")
    print(f"Predicted Class: {result_fallback['predicted_class']}")
    print(f"Confidence: {result_fallback['confidence'] * 100:.2f}%")
    print(f"NIR Analysis: {result_fallback['nir_prediction']} (confidence: {result_fallback['nir_confidence']*100:.1f}%)")
    print(f"Fallback Reason: {result_fallback['fallback_reason']}")
    print(f"Fallback Triggered: {result_fallback['fallback_triggered']}")
    print("Status: SUCCESS [OK]" if result_fallback['predicted_class'] == 'PVC' else "Status: FAILED [ERROR]")

if __name__ == "__main__":
    main()
