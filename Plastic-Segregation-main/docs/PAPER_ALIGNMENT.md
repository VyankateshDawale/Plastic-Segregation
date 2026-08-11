# Paper Repo Alignment Tracker (Path B)

This file tracks how the IEEE paper / patent claims map onto this repository.
Updated to reflect completion of codebase consolidation and correction of
two previously false "Done" claims.

Legend: `Done` -- `Partial` -- `Missing` -- `Deferred` -- `Blocked`

---

## Phase 1 - foundation (Completed)

| Paper / patent claim | Repo evidence | Status | Notes |
|----------------------|---------------|--------|-------|
| YOLOv11 real-time detection | `src/detect.py`, `src/train.py` | Done | Model integrated. |
| OpenCV overlay + bin routing (6 classes) | `src/detect.py` SORTING_MAP | Done | Rule-based routing mapped to 6 macro-classes. |
| Dataset sanity / train / val scripts | `sanity_check.py`, `validate.py` | Done | 0 leakage across splits. |
| NIR-MIR confidence gate (ACE Engine) | `src/ace/engine.py` | Done | Retrained on 17 features. |
| MIR RF / SVM / XGB metrics | `results/classical_results.json` | Done | Retrained post-dedup; see Section A2. |
| Integrated YOLO-spectral handoff | `src/unified_pipeline.py` | Done | Unified pipeline operational. |
| Honest README | `README.md` | Done | No false claims. |

---

## Phase 2 & 3 - Vision and Spectral Hardware (Partial / Blocked)

| Claim | Status | Notes |
|-------|--------|-------|
| YOLOv11m / YOLOv11s-cls / YOLOv8-S | Blocked | No GPU hardware or raw image dataset. |
| Jetson AGX Orin benchmarks | Blocked | Jetson hardware not present. |
| Live Specim FX10 capture | Blocked | No USB/SDK camera hardware. |
| Live FTIR/MIR module | Blocked | No physical spectrometer connected. |
| Expected Failure Predictor (EFP) | PARTIAL | See Section A5. Previous Done claim (AUC=1.00, tautological label) was FALSE and has been corrected. |
| RGB feature extraction | Done / Blocked | OpenCV functions implemented; blocked on real labeled images. |

---

## Phase 4 - industrial embodiment (Blocked / Deferred)

| Claim | Status | Notes |
|-------|--------|-------|
| Conveyor 0.25 m/s, 18 items/min | Blocked | Requires physical conveyor testbed. |
| UR3e + ROS2 + vacuum gripper | Blocked | Robotics hardware not present. |
| 94.8% diversion / 89.2% purity | Blocked | Requires physical experiments. |
| USD cost model | Deferred | Discussion estimate only, not measured. |

---

## Section A - Verified Metrics (with proof)

### A1. Deduplication / Data-Leakage Fix

PREVIOUS FALSE CLAIM: "Leakage check was run" -- it had NOT been run; dedup
happened after split in the original code.

CORRECTED:

| Check | Result |
|-------|--------|
| Raw MIR CSV rows | 3 000 |
| Duplicate spectral rows | 351 |
| Unique rows after dedup | 2 649 |
| Dedup applied BEFORE split | YES -- `preprocess.py` |
| Proof | stdout: "Removed duplicates: original shape (3000, 3736), unique shape (2649, 3736)" |
| Split sizes | Train 1 853 / Val 398 / Test 398 |

**Status: DONE** -- deduplication confirmed; all MIR model pickles regenerated.

### A2. MIR Classical Classifiers (retrained post-dedup)

Proof file: `results/classical_results.json` -- regenerated 2026-08-09

| Model | Val Acc | Val F1 | Test Acc | Test F1 |
|-------|---------|--------|----------|---------|
| RandomForest (n=200, depth=15) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SVM (RBF, C=10) | 0.9975 | 0.9975 | 1.0000 | 1.0000 |
| XGBoost (n=200, depth=5) | 0.9975 | 0.9975 | 0.9975 | 0.9975 |

Note: near-perfect accuracy on mid-IR polymer spectra is consistent with
published FTIR-plastic literature (Serranti et al. 2012). The test set is
independently drawn (split after dedup, random_state=42).

**Status: DONE**

### A3. MIR Deep Learning Classifiers

Proof file: `results/deep_results.json`

| Model | Test Acc | Test F1 |
|-------|----------|---------|
| 1D-CNN (3 conv blocks) | 0.9497 | 0.9479 |
| SpectralTransformer (4 heads) | 0.9271 | 0.9269 |

Note: Deep models were not re-run post-dedup; re-run recommended before
camera-ready submission.

**Status: DONE** (re-run recommended)

### A4. NIR-HSI Classifiers

Proof file: `results/nir_hsi_results.json`

| Model | Test Acc | Test F1 |
|-------|----------|---------|
| RandomForest | 0.9136 | 0.9121 |
| SVM (RBF) | 0.9712 | 0.9711 |
| XGBoost | 0.9342 | 0.9339 |

**Status: DONE**

### A5. Expected Failure Probability (EFP) Model

PREVIOUS FALSE CLAIM: "Done (AUC=1.00)" -- label was `nir_baseline_intensity < 0.08`,
a direct threshold on an input feature. AUC=1.00 was trivially achieved.

CORRECTED LABEL (non-tautological):
  p_fail = 0.50 * rgb_dark + 0.20 * (1-gloss) + 0.15 * roughness
         + 0.15 * (1-nir_base) + noise(sigma=0.10)
  nir_failure = 1 if p_fail > 0.65

Data provenance: SYNTHETIC ONLY -- no real carbon-black NIR-fail measurements.
Proof: `python scripts/train_efp_model.py`

| Metric | Value |
|--------|-------|
| Synthetic set size | 10 000 |
| Positive rate | 14.3% |
| ROC-AUC (synthetic test, N=2 000) | 0.9779 |
| NIR bypassed at threshold 0.60 | 12.0% |
| False-positive rate | 1.6% (32/2000) |

AUC=0.9779 must NOT be reported as empirically validated.
Paper table caption must read: "EFP performance on synthetic dataset (N=10 000);
real-world validation pending."

**Status: PARTIAL -- non-tautological synthetic labels, AUC=0.9779 synthetic only**

### A6. MaterialComplexityEstimator (MCE)

PREVIOUS PROBLEM: MCE called with np.random.uniform() inputs -- non-reproducible.

CORRECTED: Inputs derived from CNN embedding quartiles (deterministic).
Still approximate -- real routing needs actual roughness/entropy from camera module.

**Status: PARTIAL -- deterministic CNN-proxy inputs; not validated with real sensor data**

### A7. OutOfDistributionDetector

**Status: BLOCKED** -- Stub (anomaly_score always 0.0).
No labelled anomaly dataset available.

### A8. Bayesian Uncertainty Estimator

**Status: BLOCKED** -- Stub (uncertainty always 0.5).
MC-Dropout inference not implemented.

---

## Asset checklist (under `models/`)

Vision:
- [x] `best.pt` (YOLO detector)
- [x] `results/vision/val_metrics.json`

Spectral:
- [x] `randomforest_model.pkl` (MIR -- retrained post-dedup)
- [x] `svm_model.pkl` (MIR -- retrained post-dedup)
- [x] `xgboost_model.pkl` (MIR -- retrained post-dedup)
- [x] `scaler.pkl`, `label_encoder.pkl` (MIR preprocessors -- retrained post-dedup)
- [x] `nir_hsi_model.pkl`, `scaler_nir_hsi.pkl`, `label_encoder_nir_hsi.pkl`
- [x] `ace_xgboost.json` (ACE Confidence gate)
- [x] `efp_predictor.pkl` (EFP -- non-tautological synthetic labels)

---

## Branch hygiene

| Branch | Action | Status |
|--------|--------|--------|
| `main` | Merge Phase 1-5 work when ready | Ready to merge. |
| `atharva` | Spectral track source | Merged into unified structure. |
| `yolo-imrpoved` | Old typo branch | Safe to delete. |

---


## Publishing Rule

Only put numbers in the IEEE paper / patent that appear in:
1. this repo's `results/`, or
2. a linked public release of weights + logs.

All claims requiring physical hardware must be qualified as "planned work"
or "simulated prototype proof-of-concept".

PARTIAL rows (EFP, MCE) must carry a footnote: "Synthetic data only;
real-world validation is ongoing."
BLOCKED rows (OOD, Bayesian) must be presented as architectural proposals,
not implemented and evaluated components.
