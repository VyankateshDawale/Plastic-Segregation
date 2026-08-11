"""
FTIR/NIR Plastic Classification Pipeline
Author: Antigravity AI Coding Assistant

This script executes the complete end-to-end Machine Learning and Deep Learning pipeline:
1. Performs exploratory data analysis (EDA) and data checks.
2. Preprocesses the spectral values using Savitzky-Golay filtering and Standard Normal Variate (SNV) normalization.
3. Splits data into Train, Validation, and Test sets.
4. Trains multiple models: RandomForest, SVM, XGBoost, 1D CNN, and Spectral Transformer.
5. Evaluates and compares all models on Test data.
6. Saves the best model, preprocessors, metrics, and plots.
"""

import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(4)

# ----------------- Custom Preprocessing & Utils -----------------

class SNVNormalizer:
    """Standard Normal Variate (SNV) Normalizer for spectral data."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        std[std == 0] = 1e-8
        return (X - mean) / std
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

# ----------------- Deep Learning Models -----------------

class Conv1DNet(nn.Module):
    def __init__(self, input_dim=3736, num_classes=6):
        super(Conv1DNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(8, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1) # shape: (batch, 1, seq_len)
        x = self.conv(x)
        x = x.squeeze(2)
        x = self.fc(x)
        return x

class SpectralTransformer(nn.Module):
    def __init__(self, input_dim=3736, num_classes=6, embed_dim=32, num_heads=2, num_layers=1, dim_feedforward=64):
        super(SpectralTransformer, self).__init__()
        self.patch_embed = nn.Conv1d(1, embed_dim, kernel_size=64, stride=64)
        self.num_patches = input_dim // 64
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.patch_embed(x)
        x = x.transpose(1, 2)
        
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.pos_drop(x)
        
        x = self.transformer(x)
        cls_output = x[:, 0]
        out = self.mlp_head(cls_output)
        return out

# ----------------- Core Pipeline Execution -----------------

def run_pipeline(csv_path, output_dir='.'):
    # Setup directories
    models_dir = os.path.join(output_dir, 'models')
    results_dir = os.path.join(output_dir, 'results')
    plots_dir = os.path.join(output_dir, 'plots')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print("==================================================")
    print("STEP 1: Data Inspection & EDA")
    print("==================================================")
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {csv_path} with shape {df.shape}")
    
    # Identify spectral vs metadata columns
    y_cols = [c for c in df.columns if 'Data(y)' in c or 'Data(Y)' in c]
    x_cols = [c for c in df.columns if 'Data(x)' in c or 'Data(X)' in c]
    metadata_cols = [c for c in df.columns if c not in y_cols and c not in x_cols]
    
    print(f"Spectral features detected: {len(y_cols)}")
    print(f"Metadata columns: {metadata_cols}")
    
    # Missing values check
    missing_cnt = df[y_cols].isnull().sum().sum()
    print(f"Missing values in spectral columns: {missing_cnt}")
    
    # Class balance check
    class_counts = df['Polymer'].value_counts()
    print("Class distribution:\n", class_counts)
    
    # Duplicates check
    X_raw = df[y_cols].values
    y_raw = df['Polymer'].values
    
    _, unique_indices = np.unique(X_raw, axis=0, return_index=True)
    unique_indices = sorted(unique_indices)
    
    X_unique = X_raw[unique_indices]
    y_unique = y_raw[unique_indices]
    print(f"Duplicates: {X_raw.shape[0] - X_unique.shape[0]} samples removed.")
    print(f"Unique sample size: {X_unique.shape[0]}")
    
    print("\n==================================================")
    print("STEP 2: Data Preprocessing & Split")
    print("==================================================")
    # Train/Val/Test Split (70/15/15)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_unique, y_unique, test_size=0.15, stratify=y_unique, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
    )
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    
    # Apply preprocessing pipeline
    print("Applying Savitzky-Golay filter & SNV...")
    # Savitzky-Golay
    X_train_sg = savgol_filter(X_train, window_length=15, polyorder=2, axis=1)
    X_val_sg = savgol_filter(X_val, window_length=15, polyorder=2, axis=1)
    X_test_sg = savgol_filter(X_test, window_length=15, polyorder=2, axis=1)
    
    # SNV
    snv = SNVNormalizer()
    X_train_snv = snv.fit_transform(X_train_sg)
    X_val_snv = snv.transform(X_val_sg)
    X_test_snv = snv.transform(X_test_sg)
    
    # Column Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_snv)
    X_val_scaled = scaler.transform(X_val_snv)
    X_test_scaled = scaler.transform(X_test_snv)
    
    # Label Encoding
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)
    
    # Save pipeline components
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.pkl'))
    
    print("\n==================================================")
    print("STEP 3: Model Training (Classical & Deep)")
    print("==================================================")
    
    results = {}
    classes = label_encoder.classes_
    
    # A. Classical Models
    classical_models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='rbf', C=10.0, probability=True, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric='mlogloss')
    }
    
    for name, model in classical_models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train_enc)
        joblib.dump(model, os.path.join(models_dir, f'{name.lower()}_model.pkl'))
        
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test_enc, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test_enc, y_pred, average='weighted')
        cm = confusion_matrix(y_test_enc, y_pred).tolist()
        
        results[name] = {
            "test_accuracy": float(acc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_f1": float(f1),
            "confusion_matrix": cm
        }
        print(f"  {name} Test Acc: {acc:.4f} | F1: {f1:.4f}")
        
    # B. Deep Learning Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Deep Learning device: {device}")
    
    X_tr_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train_enc, dtype=torch.long)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_enc, dtype=torch.long)
    X_te_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_te_t = torch.tensor(y_test_enc, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=128, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=128, shuffle=False)
    
    dl_models = {
        "1D_CNN": Conv1DNet(input_dim=len(y_cols), num_classes=len(classes)).to(device),
        "SpectralTransformer": SpectralTransformer(input_dim=len(y_cols), num_classes=len(classes)).to(device)
    }
    
    for name, model in dl_models.items():
        print(f"Training {name}...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001 if name=="1D_CNN" else 0.0005, weight_decay=1e-4)
        
        # Train 10 epochs
        for epoch in range(10):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
        # Evaluate
        model.eval()
        all_preds = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                
        acc = accuracy_score(y_test_enc, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test_enc, all_preds, average='weighted')
        cm = confusion_matrix(y_test_enc, all_preds).tolist()
        
        # Save PyTorch weights
        torch.save(model.state_dict(), os.path.join(models_dir, f'{name.lower()}_model.pt'))
        
        results[name] = {
            "test_accuracy": float(acc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_f1": float(f1),
            "confusion_matrix": cm
        }
        print(f"  {name} Test Acc: {acc:.4f} | F1: {f1:.4f}")
        
    print("\n==================================================")
    print("STEP 4: Model Selection & Results Summary")
    print("==================================================")
    # Find Best Model
    df_res = pd.DataFrame(results).T
    best_model_name = df_res['test_f1'].idxmax()
    best_metrics = results[best_model_name]
    print(f"Winner Model: {best_model_name}")
    print(f"Test Accuracy: {best_metrics['test_accuracy']:.4f}")
    print(f"Test F1-Score: {best_metrics['test_f1']:.4f}")
    
    # Save Best Model Info
    best_info = {
        "best_model_name": best_model_name,
        "test_accuracy": best_metrics["test_accuracy"],
        "test_f1": best_metrics["test_f1"]
    }
    with open(os.path.join(results_dir, 'best_model_info.json'), 'w') as f:
        json.dump(best_info, f, indent=4)
        
    # Save all results comparison
    with open(os.path.join(results_dir, 'all_model_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    # Plot Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    plot_data = []
    for m_name, m_metrics in results.items():
        plot_data.append({"Model": m_name, "Metric": "Accuracy", "Value": m_metrics["test_accuracy"]})
        plot_data.append({"Model": m_name, "Metric": "F1-Score", "Value": m_metrics["test_f1"]})
    df_plot = pd.DataFrame(plot_data)
    sns.barplot(x="Model", y="Value", hue="Metric", data=df_plot, palette="Set2")
    plt.ylim(0.8, 1.02)
    plt.title("Model Comparison on Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    # Plot Confusion Matrix of the Best Model
    plt.figure(figsize=(8, 7))
    sns.heatmap(np.array(best_metrics["confusion_matrix"]), annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix of Best Model ({best_model_name})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'best_model_confusion_matrix.png'), dpi=300)
    plt.close()
    
    print(f"Pipeline finished! Results are stored in '{output_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NIR/FTIR Plastic Classification Pipeline")
    parser.add_argument('--csv', type=str, default='FTIR-PLASTIC-c4/FTIR_PLASTIC_c4.csv', help='Path to the spectral dataset CSV')
    parser.add_argument('--out', type=str, default='.', help='Output root directory')
    args = parser.parse_args()
    run_pipeline(args.csv, args.out)
