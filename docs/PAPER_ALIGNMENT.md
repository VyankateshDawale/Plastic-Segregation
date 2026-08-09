# Paper ↔ Repo Alignment Tracker (Path B)

This file tracks how the IEEE paper / patent claims map onto this repository.
It has been updated to reflect the completion of the unified codebase consolidation.

Legend: `Done` · `Partial` · `Missing` · `Deferred` · `Blocked`

---

## Phase 1 — foundation (Completed)

| Paper / patent claim | Repo evidence | Status | Next action / Notes |
|----------------------|---------------|--------|---------------------|
| YOLOv11 real-time detection | `src/detect.py`, `src/train.py` | Done | Model is integrated. |
| OpenCV overlay + bin routing (6 classes) | `src/detect.py` SORTING_MAP | Done | Rule-based routing mapped to 6 macro-classes. |
| Dataset sanity / train / val scripts | `sanity_check.py`, `validate.py` | Done | Validated: 0 leakage across splits. |
| NIR→MIR confidence gate (ACE Engine) | `src/ace/engine.py`, `src/unified_pipeline.py` | Done | Retrained on 17 features, achieves 99.8% accuracy. |
| MIR RF / SVM / XGB metrics | `Atharva_NIR/results/*`, `train_classical.py` | Done | Retrained and verified on test set (100% pure spectra). |
| Integrated YOLO→spectral handoff | `src/unified_pipeline.py` | Done | Fully unified pipeline with real model fallback. |
| Honest README (no Flask/Mongo) | `README.md` | Done | Kept up-to-date with actual status. |

---

## Phase 2 & 3 — Vision and Spectral Hardware (Status: Partial / Blocked)

| Claim | Status | Build / Handoff Notes |
|-------|--------|-----------------------|
| YOLOv11m / YOLOv11s-cls / YOLOv8-S | Blocked | [BLOCKED] No raw image files or GPU hardware to train/validate. |
| Jetson AGX Orin benchmarks | Blocked | [BLOCKED] Jetson hardware not present. |
| Live Specim FX10 capture | Blocked | [BLOCKED] No USB/SDK camera hardware available. |
| Live FTIR/MIR module | Blocked | [BLOCKED] No physical spectrometer hardware connected. |
| Expected Failure Predictor (EFP) | Done | Retrained Random Forest EFP model on `nir_baseline_intensity < 0.08` (AUC 1.00) and integrated. |
| RGB-derived inputs extraction | Done / Blocked | OpenCV feature extraction functions implemented in `ExpectedFailurePredictor` class; [BLOCKED] on real validation due to missing labeled images. |

---

## Phase 4 — industrial embodiment (Status: Blocked / Deferred)

| Claim | Status | Build notes |
|-------|--------|-------------|
| Conveyor 0.25 m/s, spacing, 18 items/min | Blocked | [BLOCKED] Requires physical conveyor testbed. |
| UR3e + ROS2 + vacuum gripper | Blocked | [BLOCKED] Robotics/hardware integration not present. |
| 94.8% diversion / 89.2% purity | Blocked | [BLOCKED] Requires physical experiments. |
| USD cost model | Deferred | Keep as discussion estimate, not measured. |

---

## Asset checklist (under `models/`)

Vision:
- [x] `best.pt` (detector)
- [x] `results/vision/val_metrics.json` (from `validate.py`)

Spectral (All verified in unified repository):
- [x] `randomforest_model.pkl` (MIR classifier)
- [x] `scaler.pkl`, `label_encoder.pkl` (MIR preprocessors)
- [x] `nir_hsi_model.pkl`, `scaler_nir_hsi.pkl`, `label_encoder_nir_hsi.pkl` (NIR HSI preprocessors & classifier)
- [x] `ace_xgboost.json` (ACE Confidence gate model)
- [x] `efp_predictor.pkl` (Expected failure prediction model)

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
1. this repo’s `results/`, or
2. a linked public release of weights + logs.

All claims requiring physical hardware or raw datasets that are not present must be qualified in the draft as "planned work" or "simulated prototype proof-of-concept".
