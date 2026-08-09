import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def perform_eda():
    print("Starting Exploratory Data Analysis (EDA)...")
    
    # 1. Load dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    csv_path = os.path.join(project_root, 'FTIR-PLASTIC-c4', 'FTIR_PLASTIC_c4.csv')
    if not os.path.exists(csv_path):
        csv_path = 'FTIR-PLASTIC-c4/FTIR_PLASTIC_c4.csv'
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset from {csv_path} with shape: {df.shape}")
    
    # 2. Check for missing values
    total_missing = df.isnull().sum().sum()
    print(f"Total missing values: {total_missing}")
    
    # 3. Check for duplicates
    # Let's check duplicates based on the y-values (spectral features)
    y_cols = ['Data(y)'] + [f'Data(y).{i}' for i in range(1, 3736)]
    duplicate_rows = df.duplicated(subset=y_cols).sum()
    print(f"Number of duplicate samples based on spectral features: {duplicate_rows}")
    
    # 4. Class imbalance check
    class_counts = df['Polymer'].value_counts()
    print("\nClass distribution:")
    print(class_counts)
    
    # 5. Extract wavenumbers
    x_cols = ['Data(x)'] + [f'Data(x).{i}' for i in range(1, 3736)]
    # Use the first row to get the wavenumbers (since they are constant across rows)
    wavenumbers = df[x_cols].iloc[0].values
    
    # 6. Save EDA stats to JSON
    stats = {
        "num_samples": int(df.shape[0]),
        "num_features": len(y_cols),
        "total_missing": int(total_missing),
        "duplicate_samples": int(duplicate_rows),
        "classes": list(class_counts.index),
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "wavenumber_range": [float(wavenumbers.min()), float(wavenumbers.max())]
    }
    
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'eda_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Saved statistics to {results_dir}/eda_stats.json")
    
    # Set aesthetics for plotting
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 18
    })
    
    # 7. Plot class distribution
    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("muted", len(class_counts))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette=colors, hue=class_counts.index, legend=False)
    plt.title("Class Distribution of Plastic Polymers")
    plt.xlabel("Polymer Type")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'), dpi=300)
    plt.close()
    print("Saved class distribution plot to plots/class_distribution.png")
    
    # 8. Plot mean spectra with standard deviation bands for each class
    plt.figure(figsize=(12, 8))
    
    unique_polymers = df['Polymer'].unique()
    palette = sns.color_palette("Set1", len(unique_polymers))
    
    for idx, polymer in enumerate(unique_polymers):
        class_df = df[df['Polymer'] == polymer][y_cols]
        mean_spectrum = class_df.mean(axis=0).values
        std_spectrum = class_df.std(axis=0).values
        
        plt.plot(wavenumbers, mean_spectrum, label=polymer, color=palette[idx], linewidth=1.5)
        plt.fill_between(wavenumbers, 
                         mean_spectrum - std_spectrum, 
                         mean_spectrum + std_spectrum, 
                         color=palette[idx], 
                         alpha=0.15)
        
    plt.title("Mean FTIR Spectra of Plastic Polymers (with ±1 Std. Dev. Bands)", pad=20)
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity / Absorbance (A.U.)")
    plt.xlim(wavenumbers.max(), wavenumbers.min())  # Standard FTIR convention: high to low wavenumbers
    plt.legend(title="Polymer Type", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mean_spectra.png'), dpi=300)
    plt.close()
    print("Saved mean spectra plot to plots/mean_spectra.png")
    
    # 9. Plot correlation matrix for a subset of features to see spectral redundancy
    plt.figure(figsize=(10, 8))
    # Select 20 evenly spaced wavelengths to avoid massive correlation matrix plotting
    step = len(y_cols) // 20
    subset_cols = [y_cols[i] for i in range(0, len(y_cols), step)]
    # Label the correlation matrix using actual wavenumbers
    subset_wavenumbers = [f"{wavenumbers[i]:.1f}" for i in range(0, len(y_cols), step)]
    
    corr_matrix = df[subset_cols].corr()
    sns.heatmap(corr_matrix, xticklabels=subset_wavenumbers, yticklabels=subset_wavenumbers, 
                cmap="coolwarm", annot=True, fmt=".2f", cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Correlation Matrix of Representative Spectral Bands (Wavenumbers in cm⁻¹)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'spectral_correlation.png'), dpi=300)
    plt.close()
    print("Saved spectral correlation heatmap to plots/spectral_correlation.png")
    print("EDA completed successfully!")

if __name__ == "__main__":
    perform_eda()
