# Plastic-Segregation

Multimodal waste segregation system aligned with the project paper/patent architecture:

1. **Vision (YOLOv11 + OpenCV)** — real-time macro waste detection and bin routing  
2. **NIR spectral typing** — polymer classification for items labeled Plastic  
3. **MIR confidence-gated fallback** — used when NIR confidence/reflectance is inconclusive (e.g. black plastics)

> **Status:** Phase 1 of repo↔paper alignment. Working modules exist for (1) and offline (2)(3). Live sensors, robotics, and conveyor deployment are tracked as Phase 2–4 gaps (see [`docs/PAPER_ALIGNMENT.md`](docs/PAPER_ALIGNMENT.md)).

---

## Architecture (target = papers)

```
Webcam
  → YOLOv11 detect (6 macro classes) + OpenCV overlay
      → Non-plastic classes → bin recommendation (Green/Blue/Yellow)
      → Plastic → NIR SVM (232-band HSI mean spectrum)
           → if conf ≥ 85% AND reflectance ≥ 0.08 → accept NIR class
           → else → MIR Random Forest (3736-band FTIR) → final resin class
```

| Stage | Paper name | Repo status |
|-------|------------|-------------|
| Vision detect + OpenCV UI | Stage 1 localization / UI | **Done** — `src/detect.py` |
| YOLO train / validate | Vision training | **Done** — `src/train.py`, `src/validate.py` |
| NIR → MIR gated fallback | Spectral stages | **Done (offline)** — `Atharva_NIR/` |
| Multimodal handoff script | Integrated pipeline | **Scaffold** — `src/pipeline_multimodal.py` |
| Dual YOLO classifiers / robot / conveyor / Jetson | Later paper stages | **Not built** — see alignment doc |

---

## Repository layout

```
src/
  detect.py                 # OpenCV + YOLOv11 live detection UI
  train.py                  # Train YOLOv11s detector
  validate.py               # mAP validation
  sanity_check.py           # Dataset integrity checks
  pipeline_multimodal.py    # Plastic → NIR/MIR handoff (Phase 1 integration)
Atharva_NIR/
  inference/inference.py    # DualStagePlasticInference (85% / 0.08 gates)
  train_and_evaluate_pipeline.py
  results/                  # Metrics used in paper spectral tables
models/                     # Place best.pt and *.pkl here (gitignored binaries)
docs/PAPER_ALIGNMENT.md     # Claim → evidence → gap tracker
scripts/export_paper_metrics.py
```

---

## Quick start

### 1) Environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
# PyTorch with CUDA (example for CUDA 12.4):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2) Vision track (YOLO + OpenCV)

Place trained weights at one of:

- `models/best.pt`
- `runs/detect/train/weights/best.pt`

```bash
python src/sanity_check.py          # optional, needs dataset/
python src/train.py                 # needs dataset/data.yaml
python src/validate.py --weights models/best.pt
python src/detect.py --weights models/best.pt
```

**Vision classes:** Biodegradable, Cardboard, Glass, Metal, Paper, Plastic → OpenCV bin panel.

### 3) Spectral track (NIR / MIR)

```bash
# Requires local datasets (not in git):
#   FTIR-PLASTIC-c4/FTIR_PLASTIC_c4.csv
#   ADVANCED_PLASTIC_FUSE.rar  (HSI)
# Requires trained pickles under models/ (see Atharva_NIR/inference)

python Atharva_NIR/train_and_evaluate_pipeline.py --csv FTIR-PLASTIC-c4/FTIR_PLASTIC_c4.csv --out Atharva_NIR
python Atharva_NIR/test_inference_demo.py
```

Committed metrics (no retrain needed to inspect paper numbers):

```bash
python scripts/export_paper_metrics.py
```

### 4) Multimodal handoff (Phase 1)

```bash
python src/pipeline_multimodal.py --image path/to/frame.jpg --weights models/best.pt
# Optional spectral vectors:
python src/pipeline_multimodal.py --image frame.jpg --weights models/best.pt --nir sample_nir.npy --mir sample_mir.npy
```

If the vision model predicts `Plastic` and NIR/MIR inputs + models are available, the confidence-gated spectral stage runs.

---

## Results already in repo (spectral)

| Model | Test accuracy | Source |
|-------|---------------|--------|
| NIR SVM | 97.12% | `Atharva_NIR/results/nir_hsi_results.json` |
| MIR Random Forest | 100% | `Atharva_NIR/results/model_comparison.csv` |
| MIR XGBoost | 99.75% | same |
| MIR 1D CNN | 94.97% | same |
| MIR Spectral Transformer | 92.71% | same |

Vision mAP/FPS: export from your local `validate.py` / live `detect.py` run and place under `results/vision/` (see alignment doc).

---

## What is intentionally not claimed as done

- Dual-stage YOLOv11m + YOLOv11s-cls polymer classifier  
- UR3e / ROS2 robotic diversion  
- Conveyor throughput / sorting purity study  
- Jetson AGX Orin deployment  
- Live Specim FX10 / FTIR USB drivers  

These are **Phase 2–4** build targets so the repo can grow toward the full paper/patent embodiment.

---

## Citation / project

Vishwakarma University — Plastic Segregation multimodal project.  
Remote: https://github.com/VyankateshDawale/Plastic-Segregation


## Branch map

| Branch | Use |
|--------|-----|
| main | Project README + baseline YOLO tree |
| tharva | NIR/MIR spectral code, metrics, integration scaffolding |
| yolo-imrpoved | Vision/YOLO improvement scaffolding |
