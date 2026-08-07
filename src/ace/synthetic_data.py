import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

class SyntheticDataGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        self.polymer_classes = ['PET', 'HDPE', 'PVC', 'LDPE', 'PP', 'PS']
        
    def generate(self, n_samples=10000) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.seed)
        
        is_cb = rng.rand(n_samples) < 0.15
        
        reflectance_mean = np.zeros(n_samples)
        nir_confidence = np.zeros(n_samples)
        
        # CB samples
        cb_mask = is_cb
        if np.any(cb_mask):
            reflectance_mean[cb_mask] = rng.uniform(0.01, 0.08, np.sum(cb_mask))
            nir_confidence[cb_mask] = rng.uniform(0.05, 0.40, np.sum(cb_mask))
        
        # Non-CB samples
        ncb_mask = ~is_cb
        if np.any(ncb_mask):
            reflectance_mean[ncb_mask] = rng.uniform(0.10, 0.45, np.sum(ncb_mask))
            nir_confidence[ncb_mask] = rng.uniform(0.60, 0.99, np.sum(ncb_mask))
        
        # Environment
        ambient_light = np.clip(rng.normal(500, 150, n_samples), 50, 1200)
        sensor_temp = np.clip(rng.normal(28, 5, n_samples), 10, 50)
        humidity = np.clip(rng.normal(55, 15, n_samples), 20, 95)
        conveyor_speed = rng.uniform(0.15, 0.40, n_samples)
        
        # Noise additions
        low_light = ambient_light < 200
        high_temp = sensor_temp > 40
        high_humidity = humidity > 80
        
        if np.any(low_light):
            nir_confidence[low_light] += rng.uniform(-0.15, -0.05, np.sum(low_light))
        if np.any(high_temp):
            nir_confidence[high_temp] += rng.uniform(-0.08, -0.03, np.sum(high_temp))
        if np.any(high_humidity):
            nir_confidence[high_humidity] += rng.uniform(-0.05, -0.02, np.sum(high_humidity))
        nir_confidence = np.clip(nir_confidence, 0.0, 1.0)
        
        # Other features
        polymer_idx = rng.randint(0, 6, n_samples)
        
        nir_margin = nir_confidence * rng.uniform(0.1, 0.5, n_samples)
        reflectance_var = reflectance_mean * rng.uniform(0.01, 0.1, n_samples)
        
        spectral_entropy = np.zeros(n_samples)
        if np.any(cb_mask):
            spectral_entropy[cb_mask] = rng.uniform(1.0, 3.0, np.sum(cb_mask))
        if np.any(ncb_mask):
            spectral_entropy[ncb_mask] = rng.uniform(4.0, 6.0, np.sum(ncb_mask))
        
        obj_hue = rng.uniform(0, 180, n_samples)
        obj_val = np.zeros(n_samples)
        if np.any(cb_mask):
            obj_val[cb_mask] = rng.uniform(0, 30, np.sum(cb_mask))
        if np.any(ncb_mask):
            obj_val[ncb_mask] = rng.uniform(80, 255, np.sum(ncb_mask))
        
        obj_size = rng.uniform(500, 5000, n_samples)
        obj_ar = rng.uniform(0.5, 2.0, n_samples)
        recent_mir = rng.uniform(0.0, 0.3, n_samples)
        
        # Label logic
        mir_needed = np.zeros(n_samples, dtype=bool)
        mir_needed |= (reflectance_mean < 0.08)
        mir_needed |= (nir_confidence < 0.50)
        
        is_pvc = (polymer_idx == 2)
        mir_needed |= (is_pvc & (nir_confidence < 0.75))
        
        X = np.column_stack([
            nir_confidence,
            nir_margin,
            reflectance_mean,
            reflectance_var,
            spectral_entropy,
            obj_hue,
            obj_val,
            obj_size,
            obj_ar,
            ambient_light,
            sensor_temp,
            humidity,
            conveyor_speed,
            polymer_idx,
            recent_mir
        ])
        y = mir_needed.astype(int)
        
        return X, y

    def generate_dataframe(self, n_samples=10000):
        if pd is None:
            raise ImportError("pandas is required for generate_dataframe")
            
        X, y = self.generate(n_samples)
        
        from .feature_extractor import FeatureExtractor
        cols = FeatureExtractor().feature_names()
        
        df = pd.DataFrame(X, columns=cols)
        df['MIR_needed'] = y
        return df
