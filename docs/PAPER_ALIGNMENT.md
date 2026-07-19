# Paper ↔ Repo Alignment Tracker (Path B)

This file tracks how the IEEE paper / patent claims map onto this repository.
Update the **Status** column as Phase 2–4 work lands. Do not mark Done without
committed code + reproducible metrics.

Legend: `Done` · `Partial` · `Missing` · `Deferred`

---

## Phase 1 (current) — foundation

| Paper / patent claim | Repo evidence | Status | Next action |
|----------------------|---------------|--------|-------------|
| YOLOv11 real-time detection | `src/detect.py`, `src/train.py` | Done | Commit/link `models/best.pt` |
| OpenCV overlay + bin routing (6 classes) | `src/detect.py` SORTING_MAP | Done | Add screenshots under `results/vision/` |
| Dataset sanity / train / val scripts | `sanity_check.py`, `validate.py` | Done | Run validate; save mAP table |
| NIR→MIR confidence gate (85% / 0.08) | `Atharva_NIR/inference/inference.py` | Done | Ensure pickles in `models/` |
| MIR RF / SVM / XGB / CNN metrics | `Atharva_NIR/results/*` | Done | `python scripts/export_paper_metrics.py` |
| Integrated YOLO→spectral handoff | `src/pipeline_multimodal.py` | Partial | Needs weights + optional .npy spectra |
| Honest README (no Flask/Mongo) | `README.md` | Done | Keep updated as phases finish |

---

## Phase 2 — vision claims in the papers

| Claim | Status | Build notes |
|-------|--------|-------------|
| YOLOv11m (vs current default YOLOv11s) | Missing | Retrain with `--model yolo11m.pt`; save run metrics |
| Dual-stage detect + cls (YOLOv11s-cls) | Missing | Only add if classes are real waste types — **do not** call Clothes/Shoes “polymers” |
| YOLOv11n-cls non-plastic 5-bin model | Missing | Optional; or keep rule-based routing from Stage-1 classes |
| YOLOv8-S binary plastic/non-plastic | Missing | Optional; Plastic class from current detector may suffice |
| Reported 95.7% Top-1 / latency tables | Missing | Produce from real training logs only |
| Jetson AGX Orin benchmarks | Missing | Requires Jetson hardware + export ONNX/TensorRT |

---

## Phase 3 — live spectral hardware

| Claim | Status | Build notes |
|-------|--------|-------------|
| Live Specim FX10 capture | Missing | SDK/USB capture → 232-vector → inference |
| Live FTIR/MIR module | Missing | Capture → 3736-vector → RF |
| Real carbon-black sample study | Missing | Replace simulated `×0.04` demo with measured CB spectra + logged confidences |
| ROC calibration (0.4% / 3.2%) | Missing | Script on validation set; commit ROC plot + numbers |
| Shannon-entropy MIR trigger (patent Claim 7) | Missing | Implement in `inference.py` only if validated |

---

## Phase 4 — industrial embodiment (patent/paper full system)

| Claim | Status | Build notes |
|-------|--------|-------------|
| Conveyor 0.25 m/s, spacing, 18 items/min | Missing | Needs physical testbed + timing logs |
| UR3e + ROS2 + vacuum gripper | Missing | Major robotics workstream |
| 94.8% diversion / 89.2% purity | Missing | Only after Phase 4 experiments |
| USD cost model | Deferred | Keep as estimate in paper Discussion, not a “measured” result |

---

## Asset checklist (put under `models/`)

Vision:

- [ ] `best.pt` (detector)
- [ ] `results/vision/val_metrics.json` (from `validate.py`)
- [ ] `results/vision/demo_screenshot.jpg`

Spectral:

- [ ] `randomforest_model.pkl`
- [ ] `scaler.pkl`, `label_encoder.pkl`
- [ ] `nir_hsi_model.pkl`, `scaler_nir_hsi.pkl`, `label_encoder_nir_hsi.pkl`
- [ ] Optional sample `nir_232.npy` / `mir_3736.npy` for CI smoke test

---

## Branch hygiene

| Branch | Action |
|--------|--------|
| `main` | Merge Phase 1 work when ready (YOLO + docs); keep as default |
| `atharva` | Spectral track source — merge into `main` after Phase 1 review |
| `yolo-imrpoved` | Identical to old main — **delete after merge** (typo branch) |

---

## Publishing rule

Only put numbers in the IEEE paper / patent that appear in:

1. this repo’s `results/`, or  
2. a linked public release of weights + logs  

If a paper number is not reproducible from those, mark the claim `Missing` here and remove/qualify it in the draft.
