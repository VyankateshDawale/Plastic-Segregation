# 🔬 Waste Segregation System — Improvement Plan for Publishability & Patentability

> **Project:** Multimodal AI-Driven Waste Segregation System  
> **Date:** 2026-07-10  
> **Status:** PLANNING — Awaiting Approval  

---

## Current State Assessment

| Component | Current Metric | Target | Gap |
|---|---|---|---|
| YOLOv11s Detection mAP@50 | 55.1% | >85% | **Critical** |
| YOLOv11s-cls Top-1 Accuracy | 95.7% | 95%+ | ✅ Already good |
| NIR SVM Accuracy | 97.12% | 97%+ | ✅ Already good |
| MIR RF Accuracy | 100% (lab) | Needs honesty caveat | ⚠️ Credibility risk |
| Dataset Size (Detection) | 7,324 train / 2,098 val / 1,042 test | 15,000+ | Moderate |
| Robotic Sorting | Rule-based (HSV + weight) | ML-based | Moderate |
| Sensor Fusion | Sequential cascade | Deep fusion option | Enhancement |
| Edge Benchmarks | None | Jetson/RPi numbers | Missing |

### Hardware Available
- **GPU:** NVIDIA RTX 4050 Laptop (6 GB VRAM)
- **CPU:** Intel i7-13620H (16 threads)
- **Current model files:** `yolo11s.pt`, `yolo11s-cls.pt`, `yolo26n.pt`

---

## Phase 1: Critical Fixes (P0) — Detection mAP & Paper Integrity

> **Goal:** Get detection mAP@50 from 55.1% → 85%+ and fix paper credibility issues.  
> **Timeline:** ~4-6 hours of GPU training + code changes

### 1.1 Upgrade Detection Model: YOLOv11s → YOLOv11m

The 55.1% mAP is the single biggest barrier to publication. Root causes:
- **Model too small:** YOLOv11s has limited capacity for 6 heterogeneous classes
- **Weak augmentation:** Current config uses basic HSV/flip only, no `copy_paste`, low `mixup`
- **Low image resolution:** Training at 640px; waste objects are often small
- **Dataset quality:** Roboflow export may have loose bounding boxes

**Action Items:**

#### [MODIFY] [train.py](file:///D:/projects/Plastic%20Waste%20Sorting%20System/src/train.py)
- Change default model from `yolo11s.pt` → `yolo11m.pt`
- Increase image size from 640 → 800 (fits within 6GB VRAM for YOLOv11m at batch 4)
- Aggressive augmentation: `mosaic=1.0`, `mixup=0.15`, `copy_paste=0.1`
- Add `close_mosaic=20` to disable mosaic in final 20 epochs for fine-tuning
- Increase epochs from 100 → 200 with patience=30
- Report mAP@50-95 alongside mAP@50

#### [NEW] `src/train_detect_v2.py`
- New training script with the upgraded configuration
- Includes automatic download of `yolo11m.pt` if not present
- Saves results to `runs/detect_v2/`

#### [MODIFY] [validate.py](file:///D:/projects/Plastic%20Waste%20Sorting%20System/src/validate.py)
- Generate and save confusion matrix plot
- Report mAP@50 AND mAP@50-95
- Save per-class AP breakdown as CSV

### 1.2 Dataset Augmentation & Cleaning

**Offline augmentation script to expand training set ~2x:**

#### [NEW] `src/augment_detection_dataset.py`
- Apply synthetic occlusions (random rectangular patches)
- Motion blur simulation (conveyor-belt realism)
- Random lighting variation (brightness, contrast)
- Background diversification 
- Generate ~7,000 additional augmented images → total ~15,000 training images

### 1.3 Paper Credibility Fixes

#### [MODIFY] The IEEE paper (will regenerate after training)
- **Acknowledge 100% MIR accuracy limitation:** Add explicit caveat that the MIR RF achieved 100% on a curated lab-grade FTIR dataset, and real-world performance with contaminated/weathered/wet plastics is expected to decrease
- **Tone down novelty claims:** Rewrite intro to say "While multimodal sensor fusion has been explored [citations], no prior system has implemented an automated, confidence-gated NIR→MIR escalation mechanism driven by a real-time YOLO vision pipeline."
- **Add per-class confusion matrices** for all classifiers
- **Report mAP@50-95** alongside mAP@50

---

## Phase 2: Experimental Strengthening (P1)

> **Goal:** Add ablation studies, confusion matrices, and replace rule-based sorting.  
> **Timeline:** ~2-3 hours of coding + training

### 2.1 Generate Confusion Matrices & ROC Curves

#### [NEW] `src/generate_evaluation_plots.py`
- Run each model on its test set and generate:
  - **Confusion matrix** (normalized) for detection model
  - **Confusion matrix** for classification model (10-class)
  - **ROC curve** showing the 85% NIR confidence threshold calibration
  - **Per-class precision-recall curves**
- Save all plots as high-res PNGs in `outputs/evaluation/`

### 2.2 Ablation Studies

#### [NEW] `src/run_ablation.py`
- Compare system performance under different configurations:
  1. **Vision-only** (YOLOv11s-cls alone, no spectral)
  2. **NIR-only** (SVM, no fallback)
  3. **MIR-only** (RF alone)
  4. **NIR + MIR sequential** (current system)
  5. **Fixed threshold (85%)** vs alternative thresholds (70%, 80%, 90%, 95%)
- Output: comparison table as CSV + formatted table for paper

### 2.3 Replace Rule-Based Robotic Sorting (Paper Enhancement)

The rule-based HSV + weight classifier is inconsistent with the AI-driven theme.

#### [NEW] `src/train_nonplastic_cls.py`
- Train a lightweight YOLOv11n-cls model on non-plastic categories (Paper, Metal, Glass, Organic, Other)
- Uses crops from the existing detection model's output
- This gives us a proper ML-based sorting for the robotic arm stage
- Report confusion matrix for non-plastic categories

---

## Phase 3: Strategic Enhancements (P2)

> **Goal:** Strengthen patent position and add differentiating features.  
> **Timeline:** Longer-term, after P0/P1 are complete

### 3.1 Cross-Modal Attention Fusion (Patent Differentiator)

#### [NEW] `src/fusion/cross_modal_attention.py`
- Implement a lightweight cross-modal attention mechanism
- Architecture: Bidirectional Cross-Attention between RGB features (from YOLO backbone) and spectral features (NIR/MIR vectors)
- Purpose: Learn to dynamically weight modalities rather than hard-coding sequential cascade
- This is a **second patentable contribution** beyond the confidence-gated escalation

### 3.2 Edge Deployment Benchmarking Scripts

#### [NEW] `src/benchmark_edge.py`
- Script to benchmark inference on current hardware (RTX 4050 laptop)
- Export models to ONNX and TensorRT for optimized inference
- Measure and report: FPS, latency per stage, GPU memory, power draw
- Output: formatted benchmark table for paper

> [!WARNING]
> **Jetson/RPi benchmarks require physical hardware.** The script can be written now but must be run on those devices when available. For the paper, we can report the RTX 4050 laptop numbers and state "Jetson deployment is planned for future work."

### 3.3 Dataset Public Release Preparation

#### [NEW] `src/prepare_dataset_release.py`
- Package the detection + classification datasets with proper documentation
- Generate `datasheet.md` following the Datasheets for Datasets standard
- Anonymize any metadata
- Prepare for upload to Zenodo/HuggingFace

---

## Execution Order & Dependencies

```
Phase 1 (P0 — CRITICAL, DO FIRST)
├── 1.1 Upgrade detection model → train_detect_v2.py [~3-4 hours GPU]
├── 1.2 Dataset augmentation → augment_detection_dataset.py [~30 min]
└── 1.3 Paper credibility fixes → regenerate paper [after 1.1 completes]

Phase 2 (P1 — IMPORTANT, DO SECOND)
├── 2.1 Confusion matrices & ROC curves → generate_evaluation_plots.py
├── 2.2 Ablation studies → run_ablation.py
└── 2.3 Non-plastic ML classifier → train_nonplastic_cls.py

Phase 3 (P2 — ENHANCEMENT, DO LAST)
├── 3.1 Cross-modal attention fusion → fusion/cross_modal_attention.py
├── 3.2 Edge benchmarking → benchmark_edge.py
└── 3.3 Dataset release prep → prepare_dataset_release.py
```

---

## What We CAN Do Right Now vs. What Needs External Resources

### ✅ Can Do Now (Software/Training)
| Task | Effort | Impact |
|---|---|---|
| Retrain detection model (YOLOv11m, better augment) | 4-6 hrs GPU | **Massive** — mAP 55→85%+ |
| Generate confusion matrices for all models | 1 hr | **High** — paper requirement |
| Ablation study (threshold sweep, modality comparison) | 2 hrs | **High** — paper requirement |
| Add MIR accuracy caveat to paper | 15 min | **High** — credibility |
| Rewrite novelty claims | 30 min | **High** — reviewers will flag |
| Train non-plastic classifier (ML replacement) | 1-2 hrs | Medium |
| Edge benchmark on RTX 4050 | 30 min | Medium |
| ONNX/TensorRT export for speed | 1 hr | Medium |
| Cross-modal attention prototype | 3-4 hrs | Medium-High (patent) |

### ❌ Needs External Resources (Cannot Do in Code)
| Task | Requirement | Notes |
|---|---|---|
| File provisional patent | Legal counsel, USPTO filing (~$150) | Protect confidence-gated escalation ASAP |
| Joint RGB+NIR+MIR dataset | Physical hardware setup (NIR/MIR sensors + conveyor) | Most impactful for publication |
| Jetson/RPi benchmarks | Physical Jetson Nano, RPi 5 devices | Can write script now, run later |
| E-waste / multi-layer plastic expansion | New samples + FTIR scans | Future work section |
| XRF as third spectral modality | XRF sensor hardware ($10k+) | Future work section |
| Video demonstration | Record system in action with phone camera | Easy to do physically |

---

## Target Journals (Post-Fix)

| Journal | Impact | Fit | Submission Ready After |
|---|---|---|---|
| Waste Management (Elsevier) | High | ★★★★★ | Phase 1+2 complete |
| Resources, Conservation and Recycling | High | ★★★★☆ | Phase 1+2 complete |
| IEEE Trans. Industrial Informatics | High | ★★★★☆ | Phase 2+3 complete |
| IEEE Access | Medium | ★★★☆☆ | Phase 1 complete (faster review) |

---

## Summary: What I'll Build

Upon approval, I will execute **Phase 1 and Phase 2** immediately:

1. **`src/train_detect_v2.py`** — Upgraded YOLOv11m detection training with aggressive augmentation
2. **`src/augment_detection_dataset.py`** — Offline dataset augmentation (2x expansion)
3. **`src/generate_evaluation_plots.py`** — Confusion matrices, ROC curves, PR curves for all models
4. **`src/run_ablation.py`** — Full ablation study (modality comparison + threshold sweep)
5. **`src/train_nonplastic_cls.py`** — ML-based non-plastic classifier replacing rule-based sorting
6. **`src/benchmark_edge.py`** — Edge deployment benchmarking script
7. **Regenerated IEEE paper** — Updated with new mAP, confusion matrices, ablations, caveats

> [!IMPORTANT]
> **Phase 1.1 (detection retraining) will take 3-6 hours of GPU time.** I recommend starting this first and working on other tasks in parallel while it trains.
