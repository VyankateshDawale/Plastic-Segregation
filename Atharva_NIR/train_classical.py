import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import json
import os

def train_classical_models():
    print("Loading preprocessed dataset...")
    data = np.load('results/dataset_split.npz', allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    classes = data['classes']
    
    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Define models
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, 
            max_depth=15, 
            random_state=42, 
            n_jobs=-1
        ),
        "SVM": SVC(
            kernel='rbf', 
            C=10.0, 
            gamma='scale', 
            probability=True, 
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, f'models/{name.lower()}_model.pkl')
        print(f"Saved {name} to models/{name.lower()}_model.pkl")
        
        # Predict
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Evaluate Validation Set
        val_acc = accuracy_score(y_val, y_val_pred)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            y_val, y_val_pred, average='weighted'
        )
        
        # Evaluate Test Set
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average='weighted'
        )
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred).tolist()
        
        print(f"{name} Validation Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")
        print(f"{name} Test Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")
        
        results[name] = {
            "val_accuracy": float(val_acc),
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
            "val_f1": float(val_f1),
            "test_accuracy": float(test_acc),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1),
            "confusion_matrix": cm
        }
        
    # Save results to json
    with open('results/classical_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nClassical models training and evaluation completed!")

if __name__ == "__main__":
    train_classical_models()
