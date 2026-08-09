"""
train_efp_model.py — Expected Failure Probability (EFP) Model Training
=======================================================================

Trains a RandomForest model to predict BEFORE engaging the NIR spectrometer
whether NIR will fail on a given plastic item (e.g. carbon-black absorbers).

LABEL DESIGN (non-tautological):
  The true label nir_failure is derived from a noisy weighted combination of
  visual inputs — rgb_darkness, gloss_index, texture_roughness — with added
  Gaussian noise so the model has to learn a real pattern, not copy a feature.
  nir_baseline_intensity is included as an *input* feature (fast NIR point-
  scan) but is NOT the sole source of the label, preventing the tautological
  AUC=1.00 case.

Data provenance: SYNTHETIC ONLY. No real labeled NIR-fail/succeed events
  exist yet. AUC should be interpreted as synthetic-only performance.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
import joblib


def generate_synthetic_data(num_samples: int = 10000) -> pd.DataFrame:
    """Generate non-tautological synthetic EFP training data.
    
    Features:
        rgb_darkness           – proxy for carbon-black presence (0=white, 1=black)
        nir_baseline_intensity – fast pre-scan mean reflectance
        lighting_lux           – ambient lux in factory
        gloss_index            – specular highlight ratio (0=matte, 1=glossy)
        texture_roughness      – Laplacian variance proxy (0=smooth, 1=rough)
        object_size_cm         – bounding-box longest side estimate

    Label (nir_failure):
        A NOISY combination of the above features, NOT a simple threshold on
        a single input.  The model therefore learns an actual multivariate
        decision boundary.
    """
    print("Generating non-tautological synthetic dataset for EFP...")
    np.random.seed(42)

    # ---- features ---------------------------------------------------------
    rgb_darkness = np.random.beta(a=2, b=5, size=num_samples)

    # nir_baseline_intensity correlated with darkness but with noise
    nir_baseline_intensity = np.clip(
        1.0 - (rgb_darkness * np.random.uniform(0.7, 1.1, num_samples))
        + np.random.normal(0, 0.05, num_samples),
        0.01, 1.0
    )
    # Inject 15% explicit "black-plastic" samples with very low reflectance
    cb_mask = np.random.rand(num_samples) < 0.15
    rgb_darkness[cb_mask]           = np.random.uniform(0.80, 0.98, cb_mask.sum())
    nir_baseline_intensity[cb_mask] = np.random.uniform(0.01, 0.12, cb_mask.sum())

    lighting_lux      = np.random.uniform(300, 1000, size=num_samples)
    gloss_index       = np.random.uniform(0, 1,    size=num_samples)
    texture_roughness = np.random.uniform(0, 1,    size=num_samples)
    object_size_cm    = np.random.uniform(2.0, 30.0, size=num_samples)

    # ---- label ------------------------------------------------------------
    # Failure probability = weighted combination of visual cues + noise.
    # rgb_darkness and (1-gloss) are the dominant predictors of NIR absorption.
    # nir_baseline_intensity contributes too, but is NOT the sole term.
    p_fail = (
        0.50 * rgb_darkness
        + 0.20 * (1.0 - gloss_index)
        + 0.15 * texture_roughness
        + 0.15 * (1.0 - nir_baseline_intensity)   # corroborating signal, not defining it
        + np.random.normal(0, 0.10, num_samples)  # real-world noise
    )
    nir_failure = (p_fail > 0.65).astype(int)

    df = pd.DataFrame({
        'rgb_darkness':           rgb_darkness,
        'nir_baseline_intensity': nir_baseline_intensity,
        'lighting_lux':           lighting_lux,
        'gloss_index':            gloss_index,
        'texture_roughness':      texture_roughness,
        'object_size_cm':         object_size_cm,
        'nir_failure':            nir_failure
    })
    print(f"  Positive class (NIR failure): {nir_failure.sum()} / {num_samples} ({100*nir_failure.mean():.1f}%)")
    return df


def train_and_evaluate(df: pd.DataFrame) -> None:
    print("Training Expected Failure Probability model...")
    X = df.drop('nir_failure', axis=1)
    y = df['nir_failure']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_prob           = clf.predict_proba(X_test)[:, 1]
    threshold        = 0.60
    y_pred_custom    = (y_prob > threshold).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred_custom, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()
    nir_bypassed   = tp + fp
    nir_time_saved = (nir_bypassed / len(y_test)) * 100

    print("\n--- EVALUATION REPORT (SYNTHETIC DATA ONLY) ---")
    print(f"  Data source  : SYNTHETIC (non-tautological weighted combination)")
    print(f"  ROC-AUC      : {auc:.4f}")
    print(f"  NIR bypassed : {nir_bypassed} items ({nir_time_saved:.1f}%)")
    print(f"  TN (correct pass to NIR)  : {tn}")
    print(f"  FP (incorrect skip NIR)   : {fp}")
    print(f"  FN (wasted NIR time)      : {fn}")
    print(f"  TP (correct skip NIR)     : {tp}")

    # ---- save model -------------------------------------------------------
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    models_dir   = os.path.join(project_root, 'models')
    docs_dir     = os.path.join(project_root, 'docs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(docs_dir,   exist_ok=True)

    model_path = os.path.join(models_dir, 'efp_predictor.pkl')
    joblib.dump(clf, model_path)
    print(f"\n[OK] Model saved to: {model_path}")

    # ---- report -----------------------------------------------------------
    report_content = f"""# Expected Failure Probability (EFP) Model — Training Report

## Data Provenance
**SYNTHETIC DATA ONLY.** No real labeled NIR-fail/succeed events exist yet.
Metrics in this report must NOT be reported as empirically validated results
until real carbon-black sample measurements are available.

## Label Design (Non-tautological)
The NIR-failure label is derived from a noisy weighted combination of visual
inputs:

```
p_fail = 0.50 * rgb_darkness
       + 0.20 * (1 - gloss_index)
       + 0.15 * texture_roughness
       + 0.15 * (1 - nir_baseline_intensity)   # corroborating, not defining
       + noise(mu=0, sigma=0.10)
nir_failure = 1 if p_fail > 0.65 else 0
```

`nir_baseline_intensity` appears as an input feature AND contributes partially
to the label, but the dominant terms are the VISUAL features `rgb_darkness`
and `gloss_index`.  This avoids the AUC=1.00 tautology of the previous version
where the label was simply `nir_baseline_intensity < 0.08`.

## Model Inputs
| Feature | Description |
|---------|-------------|
| `rgb_darkness` | 1 - mean(gray channel) / 255, proxy for carbon-black presence |
| `nir_baseline_intensity` | Fast pre-scan mean reflectance |
| `lighting_lux` | Ambient illuminance (300–1000 lux) |
| `gloss_index` | Fraction of specular highlight pixels |
| `texture_roughness` | Normalised Laplacian variance |
| `object_size_cm` | Estimated object size in cm |

## Performance Metrics (Synthetic Test Set — 2,000 items)
- **Data source**: SYNTHETIC (non-tautological)
- **ROC-AUC Score**: {auc:.4f}
- **Decision Threshold Applied**: 0.60
- **NIR Bypassed**: {nir_bypassed} items ({nir_time_saved:.1f}%)
- **Correctly skipped NIR (TP)**: {tp}
- **Incorrectly skipped NIR (FP)**: {fp}

## Status in PAPER_ALIGNMENT.md
This row is marked **Partial — synthetic labels only, not yet validated on
real NIR-fail / NIR-succeed measurements.**
"""
    report_path = os.path.join(docs_dir, 'efp_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"[OK] Report saved to: {report_path}")


if __name__ == "__main__":
    df = generate_synthetic_data(10000)
    train_and_evaluate(df)
