import os
import numpy as np
import joblib
from scipy.signal import savgol_filter

class SNVNormalizer:
    """Standard Normal Variate (SNV) Normalizer for spectral data."""
    def transform(self, X):
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        std[std == 0] = 1e-8
        return (X - mean) / std

class DualStagePlasticInference:
    def __init__(self, models_dir='models', confidence_threshold=0.85, black_plastic_threshold=0.08):
        self.confidence_threshold = confidence_threshold
        self.black_plastic_threshold = black_plastic_threshold
        self.snv = SNVNormalizer()
        
        # 1. Load NIR HSI Model & Preprocessors (232 features)
        self.scaler_nir = joblib.load(os.path.join(models_dir, 'scaler_nir_hsi.pkl'))
        self.label_encoder_nir = joblib.load(os.path.join(models_dir, 'label_encoder_nir_hsi.pkl'))
        self.model_nir = joblib.load(os.path.join(models_dir, 'nir_hsi_model.pkl'))
        
        # 2. Load MIR Model & Preprocessors (3,736 features)
        self.scaler_mir = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        self.label_encoder_mir = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        self.model_mir = joblib.load(os.path.join(models_dir, 'randomforest_model.pkl'))
        
    def preprocess_nir(self, raw_nir):
        """Preprocesses raw NIR spectrum (232 features)."""
        X = np.atleast_2d(raw_nir)
        X_scaled = self.scaler_nir.transform(X)
        return X_scaled
        
    def preprocess_mir(self, raw_mir):
        """Preprocesses raw MIR spectrum (3,736 features)."""
        X = np.atleast_2d(raw_mir)
        X_sg = savgol_filter(X, window_length=15, polyorder=2, axis=1)
        X_snv = self.snv.transform(X_sg)
        X_scaled = self.scaler_mir.transform(X_snv)
        return X_scaled
        
    def classify_sample(self, raw_nir=None, raw_mir=None):
        """
        Executes the dual-stage fallback classification:
        1. Checks if the sample is black plastic (low NIR reflectance signal).
        2. Classifies using the NIR sensor.
        3. If low reflectance, low confidence, or organic/other class is predicted, falls back to the MIR sensor.
        """
        if raw_nir is None and raw_mir is None:
            raise ValueError("Must provide at least one sensor reading (NIR or MIR).")
            
        # Stage 1: NIR Scan
        if raw_nir is not None:
            print("[STAGE 1] Scanning with NIR sensor...")
            
            # Physical Check: Low reflectance indicating black plastic absorption
            mean_reflectance = np.mean(raw_nir)
            is_black_plastic = mean_reflectance < self.black_plastic_threshold
            
            if is_black_plastic:
                trigger_reason = f"low reflectance (mean: {mean_reflectance:.4f} < {self.black_plastic_threshold:.3f}) indicating black plastic"
                print(f"  [NIR WARNING] NIR sensor blinded due to {trigger_reason}.")
                
                # Immediately fallback to MIR
                if raw_mir is not None:
                    return self._activate_mir_fallback(raw_mir, "BLACK_PLASTIC", mean_reflectance, trigger_reason)
                else:
                    print("  [STAGE 2 WARNING] MIR backup sensor triggered but no MIR data provided.")
                    return {
                        "sensor_used": "NIR (Inconclusive - Black Plastic)",
                        "predicted_class": "UNIDENTIFIED",
                        "confidence": 0.0,
                        "probabilities": {},
                        "fallback_triggered": True,
                        "fallback_reason": trigger_reason + " (no MIR data)"
                    }
            
            # If not black plastic, run the NIR classifier
            X_nir = self.preprocess_nir(raw_nir)
            nir_pred_enc = self.model_nir.predict(X_nir)[0]
            nir_probs = self.model_nir.predict_proba(X_nir)[0]
            
            nir_class = self.label_encoder_nir.classes_[nir_pred_enc]
            nir_confidence = nir_probs[nir_pred_enc]
            
            print(f"  [NIR Prediction] Class: {nir_class} (confidence: {nir_confidence * 100:.2f}%)")
            
            # Check if fallback is triggered due to low confidence or non-plastic trash
            if nir_confidence < self.confidence_threshold or nir_class in ['ORGANIC', 'OTHER']:
                if nir_confidence < self.confidence_threshold:
                    trigger_reason = f"low confidence ({nir_confidence * 100:.1f}% < {self.confidence_threshold * 100:.0f}%)"
                else:
                    trigger_reason = f"class identified as non-plastic '{nir_class}'"
                    
                print(f"  [NIR WARNING] NIR inconclusive due to {trigger_reason}.")
                
                if raw_mir is not None:
                    return self._activate_mir_fallback(raw_mir, nir_class, nir_confidence, trigger_reason)
                else:
                    print("  [STAGE 2 WARNING] MIR backup sensor triggered but no MIR data provided.")
                    return {
                        "sensor_used": f"NIR (Inconclusive - {nir_class})",
                        "predicted_class": nir_class,
                        "confidence": nir_confidence,
                        "probabilities": {str(self.label_encoder_nir.classes_[i]): float(nir_probs[i]) for i in range(len(nir_probs))},
                        "fallback_triggered": True,
                        "fallback_reason": trigger_reason + " (no MIR data)"
                    }
            else:
                # NIR succeeded conclusively
                return {
                    "sensor_used": "NIR",
                    "predicted_class": nir_class,
                    "confidence": nir_confidence,
                    "probabilities": {str(self.label_encoder_nir.classes_[i]): float(nir_probs[i]) for i in range(len(nir_probs))},
                    "fallback_triggered": False
                }
                
        # Direct MIR scan (if no NIR is provided)
        else:
            print("[STAGE 2 DIRECT] Direct MIR sensor scan...")
            X_mir = self.preprocess_mir(raw_mir)
            mir_pred_enc = self.model_mir.predict(X_mir)[0]
            mir_probs = self.model_mir.predict_proba(X_mir)[0]
            
            mir_class = self.label_encoder_mir.classes_[mir_pred_enc]
            mir_confidence = mir_probs[mir_pred_enc]
            
            return {
                "sensor_used": "MIR (Direct)",
                "predicted_class": mir_class,
                "confidence": mir_confidence,
                "probabilities": {str(self.label_encoder_mir.classes_[i]): float(mir_probs[i]) for i in range(len(mir_probs))},
                "fallback_triggered": False
            }

    def _activate_mir_fallback(self, raw_mir, nir_class, nir_confidence, trigger_reason):
        """Helper to run the MIR backup sensor prediction."""
        print("[STAGE 2] Activating MIR backup sensor...")
        X_mir = self.preprocess_mir(raw_mir)
        mir_pred_enc = self.model_mir.predict(X_mir)[0]
        mir_probs = self.model_mir.predict_proba(X_mir)[0]
        
        mir_class = self.label_encoder_mir.classes_[mir_pred_enc]
        mir_confidence = mir_probs[mir_pred_enc]
        
        return {
            "sensor_used": "MIR (Backup)",
            "predicted_class": mir_class,
            "confidence": mir_confidence,
            "probabilities": {str(self.label_encoder_mir.classes_[i]): float(mir_probs[i]) for i in range(len(mir_probs))},
            "fallback_triggered": True,
            "nir_prediction": nir_class,
            "nir_confidence": nir_confidence,
            "fallback_reason": trigger_reason
        }

if __name__ == "__main__":
    classifier = DualStagePlasticInference()
    print("DualStagePlasticInference pipeline initialized with physical black plastic detection.")
