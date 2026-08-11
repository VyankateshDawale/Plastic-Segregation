import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ACEEngine:
    def __init__(self, model_path=None):
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            eval_metric='logloss'
        )
        self.feature_names = None
        if model_path:
            self.load(model_path)

    def fit(self, X, y, val_split=0.2) -> dict:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42)
        
        print("Training ACE Engine...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        preds = self.model.predict(X_val)
        metrics = {
            'accuracy': accuracy_score(y_val, preds),
            'precision': precision_score(y_val, preds, zero_division=0),
            'recall': recall_score(y_val, preds, zero_division=0),
            'f1': f1_score(y_val, preds, zero_division=0)
        }
        
        print(f"Validation Metrics: {metrics}")
        return metrics

    def predict_threshold(self, features) -> float:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        prob_mir_needed = self.model.predict_proba(features)[0, 1]
        
        base_threshold = 0.70
        alpha = 0.25
        threshold = base_threshold + alpha * prob_mir_needed
        return float(threshold)

    def should_escalate(self, nir_confidence, features) -> tuple[bool, float]:
        threshold = self.predict_threshold(features)
        escalate = float(nir_confidence) < threshold
        return escalate, threshold

    def feature_importance(self) -> dict:
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance = self.model.feature_importances_
        if self.feature_names:
            return {name: float(imp) for name, imp in zip(self.feature_names, importance)}
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)
