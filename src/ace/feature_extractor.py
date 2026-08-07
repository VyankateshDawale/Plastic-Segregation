import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self, polymer_classes=['PET', 'HDPE', 'PVC', 'LDPE', 'PP', 'PS']):
        self.polymer_classes = polymer_classes
        self.class_to_idx = {cls: i for i, cls in enumerate(polymer_classes)}

    def extract(self, nir_confidence, nir_margin, nir_spectrum, rgb_crop, bbox_wh, predicted_class, env_data=None, recent_history=None) -> np.ndarray:
        if env_data is None:
            env_data = {}
        
        # Spectral features
        reflectance_mean = np.mean(nir_spectrum) if nir_spectrum is not None else 0.0
        reflectance_variance = np.var(nir_spectrum) if nir_spectrum is not None else 0.0
        
        spectral_entropy = 0.0
        if nir_spectrum is not None and np.sum(nir_spectrum) > 0:
            p = nir_spectrum / np.sum(nir_spectrum)
            p = p[p > 0]
            spectral_entropy = -np.sum(p * np.log(p))

        # Visual features
        object_hue_mean = 0.0
        object_value_mean = 0.0
        if rgb_crop is not None and rgb_crop.size > 0:
            hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2HSV)
            object_hue_mean = np.mean(hsv[:, :, 0])
            object_value_mean = np.mean(hsv[:, :, 2])
            
        object_size_px = bbox_wh[0] * bbox_wh[1] if bbox_wh is not None else 0.0
        object_aspect_ratio = bbox_wh[0] / (bbox_wh[1] + 1e-5) if bbox_wh is not None else 1.0

        # Environmental
        ambient_light_lux = env_data.get('ambient_light_lux', 500.0)
        sensor_temperature_c = env_data.get('sensor_temperature_c', 25.0)
        humidity_pct = env_data.get('humidity_pct', 50.0)
        conveyor_speed_ms = env_data.get('conveyor_speed_ms', 0.25)

        # Classification
        predicted_class_encoded = self.class_to_idx.get(predicted_class, -1)
        recent_mir_rate = recent_history.get('recent_mir_rate', 0.0) if recent_history else 0.0

        features = [
            float(nir_confidence),
            float(nir_margin),
            float(reflectance_mean),
            float(reflectance_variance),
            float(spectral_entropy),
            float(object_hue_mean),
            float(object_value_mean),
            float(object_size_px),
            float(object_aspect_ratio),
            float(ambient_light_lux),
            float(sensor_temperature_c),
            float(humidity_pct),
            float(conveyor_speed_ms),
            float(predicted_class_encoded),
            float(recent_mir_rate)
        ]
        return np.array(features, dtype=np.float32)

    def feature_names(self) -> list[str]:
        return [
            'nir_confidence',
            'nir_confidence_margin',
            'reflectance_mean',
            'reflectance_variance',
            'spectral_entropy',
            'object_hue_mean',
            'object_value_mean',
            'object_size_px',
            'object_aspect_ratio',
            'ambient_light_lux',
            'sensor_temperature_c',
            'humidity_pct',
            'conveyor_speed_ms',
            'predicted_class_encoded',
            'recent_mir_rate'
        ]
