import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Set PyTorch threads for CPU efficiency
torch.set_num_threads(4)

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Network Definitions

class Conv1DNet(nn.Module):
    def __init__(self, input_dim=3736, num_classes=6):
        super(Conv1DNet, self).__init__()
        self.conv = nn.Sequential(
            # Input: (batch, 1, 3736)
            nn.Conv1d(1, 8, kernel_size=15, stride=4, padding=7), # Larger stride to downsample faster
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # (batch, 8, 467)
            nn.Conv1d(8, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # (batch, 16, 116)
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
        x = x.unsqueeze(1) # shape: (batch, 1, 3736)
        x = self.conv(x) # shape: (batch, 32, 1)
        x = x.squeeze(2) # shape: (batch, 32)
        x = self.fc(x)
        return x

class SpectralTransformer(nn.Module):
    def __init__(self, input_dim=3736, num_classes=6, embed_dim=32, num_heads=2, num_layers=1, dim_feedforward=64):
        super(SpectralTransformer, self).__init__()
        # Use a kernel_size of 64 and stride of 64 to downsample the sequence length to 58.
        # This dramatically reduces self-attention computational cost.
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
        x = x.unsqueeze(1) # shape: (batch, 1, 3736)
        x = self.patch_embed(x) # shape: (batch, embed_dim, num_patches)
        x = x.transpose(1, 2) # shape: (batch, num_patches, embed_dim)
        
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.pos_drop(x)
        
        x = self.transformer(x)
        cls_output = x[:, 0]
        out = self.mlp_head(cls_output)
        return out

# 2. Helper function to train a model
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)
            
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)
                
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
    return history

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    cm = confusion_matrix(all_targets, all_preds).tolist()
    
    return acc, precision, recall, f1, cm

def main():
    print("Loading datasets...")
    data = np.load('results/dataset_split.npz', allow_pickle=True)
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.long)
    X_val = torch.tensor(data['X_val'], dtype=torch.float32)
    y_val = torch.tensor(data['y_val'], dtype=torch.long)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Use larger batch size to speed up CPU training
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 1. Train 1D CNN
    print("\n--- Training 1D CNN ---")
    cnn_model = Conv1DNet(input_dim=3736, num_classes=6).to(device)
    cnn_history = train_model(cnn_model, train_loader, val_loader, epochs=10, lr=0.002)
    
    # Save CNN weights
    torch.save(cnn_model.state_dict(), 'models/cnn_model.pt')
    print("Saved 1D CNN state dict to models/cnn_model.pt")
    
    # Evaluate CNN
    cnn_acc, cnn_prec, cnn_rec, cnn_f1, cnn_cm = evaluate_model(cnn_model, test_loader)
    print(f"1D CNN Test Acc: {cnn_acc:.4f} | Precision: {cnn_prec:.4f} | Recall: {cnn_rec:.4f} | F1: {cnn_f1:.4f}")
    
    # 2. Train Spectral Transformer
    print("\n--- Training Spectral Transformer ---")
    transformer_model = SpectralTransformer(input_dim=3736, num_classes=6).to(device)
    transformer_history = train_model(transformer_model, train_loader, val_loader, epochs=10, lr=0.001)
    
    # Save Transformer weights
    torch.save(transformer_model.state_dict(), 'models/transformer_model.pt')
    print("Saved Spectral Transformer state dict to models/transformer_model.pt")
    
    # Evaluate Transformer
    trans_acc, trans_prec, trans_rec, trans_f1, trans_cm = evaluate_model(transformer_model, test_loader)
    print(f"Transformer Test Acc: {trans_acc:.4f} | Precision: {trans_prec:.4f} | Recall: {trans_rec:.4f} | F1: {trans_f1:.4f}")
    
    # 3. Save Deep Learning Results
    deep_results = {
        "1D_CNN": {
            "val_accuracy": float(cnn_history["val_acc"][-1]),
            "val_loss": float(cnn_history["val_loss"][-1]),
            "test_accuracy": float(cnn_acc),
            "test_precision": float(cnn_prec),
            "test_recall": float(cnn_rec),
            "test_f1": float(cnn_f1),
            "confusion_matrix": cnn_cm,
            "history": cnn_history
        },
        "SpectralTransformer": {
            "val_accuracy": float(transformer_history["val_acc"][-1]),
            "val_loss": float(transformer_history["val_loss"][-1]),
            "test_accuracy": float(trans_acc),
            "test_precision": float(trans_prec),
            "test_recall": float(trans_rec),
            "test_f1": float(trans_f1),
            "confusion_matrix": trans_cm,
            "history": transformer_history
        }
    }
    
    with open('results/deep_results.json', 'w') as f:
        json.dump(deep_results, f, indent=4)
    print("Saved deep learning results to results/deep_results.json")
    
    # 4. Plot and Save Training Curves
    plt.figure(figsize=(14, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(cnn_history["train_loss"], label="CNN Train Loss", color='blue', linestyle='--')
    plt.plot(cnn_history["val_loss"], label="CNN Val Loss", color='blue')
    plt.plot(transformer_history["train_loss"], label="Transformer Train Loss", color='green', linestyle='--')
    plt.plot(transformer_history["val_loss"], label="Transformer Val Loss", color='green')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(cnn_history["train_acc"], label="CNN Train Acc", color='blue', linestyle='--')
    plt.plot(cnn_history["val_acc"], label="CNN Val Acc", color='blue')
    plt.plot(transformer_history["train_acc"], label="Transformer Train Acc", color='green', linestyle='--')
    plt.plot(transformer_history["val_acc"], label="Transformer Val Acc", color='green')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/deep_learning_curves.png', dpi=300)
    plt.close()
    print("Saved training curves to plots/deep_learning_curves.png")

if __name__ == "__main__":
    main()
