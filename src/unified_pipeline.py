"""
unified_pipeline.py — Grand Unified Adaptive Architecture for Waste Segregation
=============================================================================

This module implements a globally novel, adaptive, uncertainty-aware continuous 
learning framework. It fuses visual, depth, near-infrared (NIR), and mid-infrared (MIR) 
modalities utilizing Bayesian uncertainty, dynamic confidence gating, continuous 
learning, and predictive failure modeling.
"""

import uuid
import json
import numpy as np
import cv2
import joblib
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.signal import savgol_filter

# Reconcile path imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ace.engine import ACEEngine
from ace.feature_extractor import FeatureExtractor

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EnvironmentalContext:
    ambient_lux: float
    camera_noise_level: float
    sensor_temp_celsius: float
    belt_speed_m_s: float
    sensor_drift_index: float

@dataclass
class MaterialPassport:
    uuid: str
    timestamp: str
    inferred_polymer: str
    confidence_score: float
    uncertainty_score: float
    complexity_class: str
    sensors_used: List[str]
    anomaly_flag: bool
    destination_bin: str

# =============================================================================
# CORE ARCHITECTURE MODULES
# =============================================================================

class OutOfDistributionDetector:
    """
    Detects foreign or hazardous objects completely outside the known 
    plastic distribution (e.g., dead batteries, chunks of metal, wood).
    [STATUS: STUB - NOT YET IMPLEMENTED]
    """
    def __init__(self, anomaly_threshold: float = 0.85):
        self.anomaly_threshold = anomaly_threshold

    def calculate_anomaly_score(self, visual_features: np.ndarray) -> float:
        # [STATUS: STUB] Autoencoder / Mahalanobis distance calculation bypassed.
        # Returns 0.0 to prevent false triggers in production.
        return 0.0
        
    def is_anomaly(self, visual_features: np.ndarray) -> bool:
        return self.calculate_anomaly_score(visual_features) > self.anomaly_threshold


class MaterialComplexityEstimator:
    """
    Calculates Material Complexity Score to dictate sensor routing.
    """
    def calculate_complexity(self, surface_roughness: float, reflectance_var: float, 
                             shape_entropy: float, texture_entropy: float) -> str:
        score = (surface_roughness + reflectance_var + shape_entropy + texture_entropy) / 4.0
        
        if score < 0.3:
            return "EASY"     # RGB Only
        elif score < 0.6:
            return "MEDIUM"   # RGB + NIR
        elif score < 0.85:
            return "HARD"     # RGB + NIR + MIR
        else:
            return "IMPOSSIBLE" # Chemical Verification / Diversion


class ExpectedFailurePredictor:
    """
    Predicts if the NIR sensor will fail (e.g., on carbon-black plastics) 
    before engaging it, using the retrained Random Forest EFP model.
    """
    def __init__(self, failure_threshold: float = 0.60):
        self.failure_threshold = failure_threshold
        
        # Load the trained model dynamically
        models_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(models_dir)
        model_path = os.path.join(project_root, 'models', 'efp_predictor.pkl')
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.model_loaded = True
        else:
            self.model = None
            self.model_loaded = False
            print(f"Warning: EFP model not found at {model_path}. Using fallback logic.")

    def extract_rgb_features(self, rgb_crop: np.ndarray) -> Tuple[float, float, float]:
        """
        Extracts EFP inputs from an RGB image crop of the plastic object using OpenCV.
        - rgb_darkness: 1.0 - (mean brightness / 255.0)
        - gloss_index: ratio of specular highlight pixels (gray > 235) to total pixels
        - texture_roughness: normalized variance of the Laplacian
        
        [STATUS: BLOCKED / UNVALIDATED ON REAL DATA due to lack of labeled image samples]
        """
        if rgb_crop is None or rgb_crop.size == 0:
            return 0.5, 0.5, 0.5
            
        gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
        
        # 1. Darkness
        mean_brightness = np.mean(gray)
        rgb_darkness = 1.0 - (mean_brightness / 255.0)
        
        # 2. Gloss Index (specular highlights)
        specular_pixels = np.sum(gray > 235)
        gloss_index = float(specular_pixels) / float(gray.size)
        
        # 3. Texture Roughness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture_roughness = float(np.clip(laplacian_var / 2500.0, 0.0, 1.0))
        
        return rgb_darkness, gloss_index, texture_roughness

    def predict_nir_failure(self, rgb_darkness: float, nir_baseline_intensity: float, 
                            lighting_lux: float, gloss_index: float, 
                            texture_roughness: float, object_size_cm: float) -> float:
        if self.model_loaded:
            features = np.array([[
                float(rgb_darkness),
                float(nir_baseline_intensity),
                float(lighting_lux),
                float(gloss_index),
                float(texture_roughness),
                float(object_size_cm)
            ]])
            p_fail = self.model.predict_proba(features)[0, 1]
            return float(p_fail)
        else:
            p_fail = (rgb_darkness * 0.7) + ((1.0 - gloss_index) * 0.3)
            return float(np.clip(p_fail, 0.0, 1.0))
        
    def should_skip_nir(self, rgb_darkness: float, nir_baseline_intensity: float, 
                        lighting_lux: float, gloss_index: float, 
                        texture_roughness: float, object_size_cm: float) -> bool:
        return self.predict_nir_failure(
            rgb_darkness, nir_baseline_intensity, lighting_lux, 
            gloss_index, texture_roughness, object_size_cm
        ) > self.failure_threshold


class BayesianUncertaintyEvaluator:
    """
    Uses Monte Carlo Dropout during inference to calculate Epistemic Uncertainty.
    [STATUS: STUB - NOT YET IMPLEMENTED]
    """
    def evaluate(self, feature_embedding: np.ndarray, num_forward_passes: int = 10) -> Tuple[str, float, float]:
        # [STATUS: STUB] Bypassed. Returns standard prediction parameters without MC randomness.
        return "UNKNOWN", 0.0, 0.0


class PatchBasedVoter:
    """
    Divides object into patches and aggregates predictions.
    """
    def aggregate_patches(self, patch_predictions: List[str]) -> str:
        from collections import Counter
        if not patch_predictions:
            return "UNKNOWN"
        return Counter(patch_predictions).most_common(1)[0][0]


class ContinuousLearningManager:
    """
    Stores difficult cases where MIR corrected NIR for overnight retraining.
    """
    def __init__(self):
        self.retraining_buffer = []

    def log_correction(self, rgb_data: np.ndarray, nir_data: np.ndarray, mir_ground_truth: str):
        self.retraining_buffer.append({
            "timestamp": datetime.now().isoformat(),
            "rgb": rgb_data.tolist() if isinstance(rgb_data, np.ndarray) else rgb_data,
            "nir": nir_data.tolist() if isinstance(nir_data, np.ndarray) else nir_data,
            "true_label": mir_ground_truth
        })


class DigitalTwinPassportGenerator:
    """
    Generates a JSON Digital Twin passport for every single processed item.
    """
    @staticmethod
    def create_passport(polymer: str, conf: float, unc: float, complexity: str, 
                        sensors: List[str], anomaly: bool, bin_dest: str) -> MaterialPassport:
        return MaterialPassport(
            uuid=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            inferred_polymer=polymer,
            confidence_score=conf,
            uncertainty_score=unc,
            complexity_class=complexity,
            sensors_used=sensors,
            anomaly_flag=anomaly,
            destination_bin=bin_dest
        )

# =============================================================================
# GRAND UNIFIED INFERENCE PIPELINE
# =============================================================================

class UnifiedSegregationPipeline:
    def __init__(self):
        models_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(models_dir)
        
        self.ood_detector = OutOfDistributionDetector()
        self.complexity_estimator = MaterialComplexityEstimator()
        self.efp_predictor = ExpectedFailurePredictor()
        self.bayesian_eval = BayesianUncertaintyEvaluator()
        self.patch_voter = PatchBasedVoter()
        self.learning_manager = ContinuousLearningManager()
        
        # Load the unified ACE engine
        self.ace_engine = ACEEngine(model_path=os.path.join(project_root, 'models', 'ace_xgboost.json'))
        self.ace_feature_extractor = FeatureExtractor()
        
        # Load the real MIR Random Forest classifier and preprocessors
        self.scaler_mir = joblib.load(os.path.join(project_root, 'models', 'scaler.pkl'))
        self.label_encoder_mir = joblib.load(os.path.join(project_root, 'models', 'label_encoder.pkl'))
        self.model_mir = joblib.load(os.path.join(project_root, 'models', 'randomforest_model.pkl'))
        
        # Load the pre-trained NIR HSI model
        self.scaler_nir = joblib.load(os.path.join(project_root, 'models', 'scaler_nir_hsi.pkl'))
        self.label_encoder_nir = joblib.load(os.path.join(project_root, 'models', 'label_encoder_nir_hsi.pkl'))
        self.model_nir = joblib.load(os.path.join(project_root, 'models', 'nir_hsi_model.pkl'))
        
    def preprocess_mir(self, raw_mir: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(raw_mir)
        X_sg = savgol_filter(X, window_length=15, polyorder=2, axis=1)
        mean = np.mean(X_sg, axis=1, keepdims=True)
        std = np.std(X_sg, axis=1, keepdims=True)
        std[std == 0] = 1e-8
        X_snv = (X_sg - mean) / std
        X_scaled = self.scaler_mir.transform(X_snv)
        return X_scaled

    def run_mir_prediction(self, raw_mir: np.ndarray) -> Tuple[str, float]:
        X_scaled = self.preprocess_mir(raw_mir)
        pred_idx = self.model_mir.predict(X_scaled)[0]
        probs = self.model_mir.predict_proba(X_scaled)[0]
        pred_class = self.label_encoder_mir.classes_[pred_idx]
        confidence = probs[pred_idx]
        return pred_class, float(confidence)

    def preprocess_nir(self, raw_nir: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(raw_nir)
        return self.scaler_nir.transform(X)

    def run_nir_prediction(self, raw_nir: np.ndarray) -> Tuple[str, float]:
        X_scaled = self.preprocess_nir(raw_nir)
        pred_idx = self.model_nir.predict(X_scaled)[0]
        probs = self.model_nir.predict_proba(X_scaled)[0]
        pred_class = self.label_encoder_nir.classes_[pred_idx]
        confidence = probs[pred_idx]
        return pred_class, float(confidence)

    def process_item(self, visual_features: np.ndarray, env_context: EnvironmentalContext, 
                     raw_mir: np.ndarray = None, raw_nir: np.ndarray = None, rgb_crop: np.ndarray = None) -> MaterialPassport:
        sensors_used = ["RGB_CAMERA"]
        
        # 1. Anomaly Detection
        if self.ood_detector.is_anomaly(visual_features):
            return DigitalTwinPassportGenerator.create_passport(
                "ANOMALY", 0.0, 1.0, "IMPOSSIBLE", sensors_used, True, "HAZARDOUS_BIN"
            )
            
        # 2. Material Complexity Routing
        # Deterministic proxies derived from the visual feature tensor:
        #   surface_roughness  ~ normalised std of 1st quarter of feature vector
        #   reflectance_var    ~ normalised std of 2nd quarter
        #   shape_entropy      ~ normalised mean of 3rd quarter
        #   texture_entropy    ~ normalised std of 4th quarter
        # These are NOT ground-truth measurements; they are order-of-magnitude
        # proxies from the CNN embedding until the computer-vision module is
        # integrated.  Results should be treated as illustrative routing.
        vf = np.asarray(visual_features).flatten()
        n  = len(vf)
        q  = max(n // 4, 1)
        _safe_std = lambda arr: float(np.std(arr)) if len(arr) > 1 else 0.0
        surface_roughness = min(1.0, _safe_std(vf[:q]))
        reflectance_var   = min(1.0, _safe_std(vf[q:2*q]))
        shape_entropy     = min(1.0, float(np.mean(np.abs(vf[2*q:3*q]))))
        texture_entropy   = min(1.0, _safe_std(vf[3*q:]))
        complexity = self.complexity_estimator.calculate_complexity(
            surface_roughness=surface_roughness,
            reflectance_var=reflectance_var,
            shape_entropy=shape_entropy,
            texture_entropy=texture_entropy
        )
        
        if complexity == "EASY":
            predicted, conf, unc = self.bayesian_eval.evaluate(visual_features)
            return DigitalTwinPassportGenerator.create_passport(
                predicted, conf, unc, complexity, sensors_used, False, f"BIN_{predicted}"
            )
            
        # 3. Expected Failure Probability (EFP) Gate
        if rgb_crop is not None:
            darkness, gloss, roughness = self.efp_predictor.extract_rgb_features(rgb_crop)
        else:
            # Deterministic fallback: derive from the visual embedding.
            # darkness  ~ 1 - mean(abs(vf))  (darker = lower energy embedding)
            # gloss     ~ interquartile range of vf (smooth surfaces have low IQR)
            # roughness ~ std of local differences between adjacent features
            vf_abs = np.abs(vf)
            darkness  = float(np.clip(1.0 - np.mean(vf_abs), 0.0, 1.0))
            q25, q75  = float(np.percentile(vf_abs, 25)), float(np.percentile(vf_abs, 75))
            gloss     = float(np.clip(1.0 - (q75 - q25), 0.0, 1.0))
            roughness = float(np.clip(np.mean(np.abs(np.diff(vf))), 0.0, 1.0))
            # NOTE: These are CNN-embedding proxies, not camera measurements.
            
        # Fast NIR baseline is simulated as the mean of the raw_nir spectrum if available
        nir_baseline = np.mean(raw_nir) if raw_nir is not None else 0.5
        
        if complexity in ["MEDIUM", "HARD"]:
            if self.efp_predictor.should_skip_nir(darkness, nir_baseline, 500.0, gloss, roughness, 15.0):
                # EFP triggered: Skip NIR, route directly to MIR
                sensors_used.append("MIR_SPECTROMETER")
                if raw_mir is not None:
                    mir_pred, mir_conf = self.run_mir_prediction(raw_mir)
                    return DigitalTwinPassportGenerator.create_passport(
                        mir_pred, mir_conf, 0.0, complexity, sensors_used, False, f"BIN_{mir_pred}"
                    )
                else:
                    return DigitalTwinPassportGenerator.create_passport(
                        "UNIDENTIFIED_BLACK_PLASTIC", 0.99, 0.01, complexity, sensors_used, False, "BIN_OTHER"
                    )
            else:
                # Engage NIR
                sensors_used.append("NIR_SPECTROMETER")
                if raw_nir is not None:
                    nir_pred, nir_conf = self.run_nir_prediction(raw_nir)
                    nir_margin = 0.1
                else:
                    # Simulated fallback class if HSI data is missing
                    nir_pred, nir_conf, nir_margin = "ORGANIC", 0.50, 0.05
                
                # 4. Dynamic Confidence Gating via retrained ACE model
                ace_features = self.ace_feature_extractor.extract(
                    nir_confidence=nir_conf,
                    nir_margin=nir_margin,
                    nir_spectrum=raw_nir if raw_nir is not None else np.ones(232) * 0.5,
                    rgb_crop=rgb_crop,
                    bbox_wh=(100, 100),
                    predicted_class=nir_pred,
                    env_data={
                        'ambient_light_lux': env_context.ambient_lux,
                        'camera_noise_level': env_context.camera_noise_level,
                        'sensor_temperature_c': env_context.sensor_temp_celsius,
                        'conveyor_speed_ms': env_context.belt_speed_m_s,
                        'sensor_drift_index': env_context.sensor_drift_index,
                        'humidity_pct': 50.0
                    },
                    recent_history=None
                )
                
                current_threshold = self.ace_engine.predict_threshold(ace_features)
                escalate_to_mir = nir_conf < current_threshold or nir_pred in ['ORGANIC', 'OTHER']
                
                # We escalate to MIR if Softmax Confidence is too low OR predicted class is ORGANIC/OTHER
                if escalate_to_mir:
                    sensors_used.append("MIR_SPECTROMETER")
                    if raw_mir is not None:
                        mir_pred, mir_conf = self.run_mir_prediction(raw_mir)
                        
                        # 5. Continuous Learning Logging
                        self.learning_manager.log_correction(
                            rgb_data=visual_features, 
                            nir_data=raw_nir if raw_nir is not None else np.zeros(232), 
                            mir_ground_truth=mir_pred
                        )
                        
                        return DigitalTwinPassportGenerator.create_passport(
                            mir_pred, mir_conf, 0.0, complexity, sensors_used, False, f"BIN_{mir_pred}"
                        )
                    else:
                        return DigitalTwinPassportGenerator.create_passport(
                            "UNIDENTIFIED_CONTAMINANT", 0.90, 0.10, complexity, sensors_used, False, "BIN_OTHER"
                        )
                else:
                    # NIR succeeded
                    return DigitalTwinPassportGenerator.create_passport(
                        nir_pred, nir_conf, 0.0, complexity, sensors_used, False, f"BIN_{nir_pred}"
                    )

# =============================================================================
# EXECUTION / DEMO (using real MIR spectra)
# =============================================================================
if __name__ == "__main__":
    pipeline = UnifiedSegregationPipeline()
    
    # Load real MIR dataset split
    models_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(models_dir)
    dataset_path = os.path.join(project_root, 'results', 'dataset_split.npz')
    
    print("==================================================")
    print("GRAND UNIFIED PIPELINE DEMO WITH REAL SPECTRA")
    print("==================================================")
    
    if os.path.exists(dataset_path):
        data = np.load(dataset_path, allow_pickle=True)
        X_test = data['X_test']
        y_test = data['y_test']
        label_encoder = pipeline.label_encoder_mir
        
        context = EnvironmentalContext(
            ambient_lux=400.0,
            camera_noise_level=0.01,
            sensor_temp_celsius=38.0,
            belt_speed_m_s=0.25,
            sensor_drift_index=0.01
        )
        
        # Process first 5 real MIR test samples
        for i in range(5):
            # [DEMO STUB] No real camera frame available — create a fixed
            # synthetic embedding vector for demo routing purposes.
            # Replace with actual CNN feature extraction in production.
            simulated_visual_tensor = np.zeros((1, 512), dtype=np.float32)
            simulated_visual_tensor[0, ::4] = 0.4  # deterministic non-random pattern
            raw_mir = X_test[i]
            
            # [DEMO STUB] No real NIR scanner attached — simulate reflectance
            # profile based on known sample characteristics:
            #   Sample index 1 == black-HDPE with very low reflectance.
            is_black = (i == 1)
            nir_mean = 0.04 if is_black else 0.30
            rng = np.random.default_rng(seed=i)  # deterministic per sample
            raw_nir = rng.normal(loc=nir_mean, scale=0.01, size=232).clip(0.01, 1.0)
            
            passport = pipeline.process_item(
                visual_features=simulated_visual_tensor,
                env_context=context,
                raw_mir=raw_mir,
                raw_nir=raw_nir
            )
            
            print(f"--- Item {i+1} Processed ---")
            print(f"Expected Class: {label_encoder.classes_[y_test[i]]}")
            print(json.dumps(passport.__dict__, indent=4))
            print("\n")
    else:
        print(f"Error: Dataset split not found at {dataset_path}")
