"""
run_ablation.py — Modality & threshold ablation study
====================================================
Produces a single CSV that the paper's ablation table is built from.

Four configurations are compared:
    1. Vision-only       : YOLOv11 detector + YOLOv11-cls classifier
    2. NIR-only          : spectral SVM (NIR reflectance)  — REQUIRES hardware
    3. MIR-only          : spectral RF  (MIR / FTIR)       — REQUIRES hardware
    4. Proposed          : Vision confidence-gated NIR->MIR escalation

IMPORTANT — publication integrity:
    Only the Vision path and the Vision portion of the Proposed system are
    actually executed here. The NIR/MIR columns are left as MISSING_HARDWARE
    placeholders until trained spectral models + a joint RGB+NIR+MIR dataset
    exist (see improvement_plan.md "Needs External Resources"). Filling these
    cells with the lab-best numbers from the plan would be fabrication and is
    deliberately avoided.

Also produced:
    - NIR confidence-threshold sweep (requires NIR scores; reported as
      NA until NIR model available). We instead report the *detector*
      confidence threshold sweep on the validation set, which is real.

Usage:
    python src/run_ablation.py
    python src/run_ablation.py --detect-model runs/detect_v2/train/weights/best.pt
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
MISSING = "MISSING_HARDWARE"   # sentinel for un-runnable spectral cells

DETECT_MODEL = PROJECT_ROOT / "runs" / "detect_v2" / "train" / "weights" / "best.pt"
OLD_DETECT = PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "train" / "weights" / "best.pt"
CLS_MODEL = PROJECT_ROOT / "runs" / "classify" / "train" / "weights" / "best.pt"
DATA_YAML = PROJECT_ROOT / "dataset" / "GARBAGE CLASSIFICATION" / "data.yaml"
CLS_DATA = PROJECT_ROOT / "dataset_cls"


def evaluate_detector(model_path: Path, data_yaml: Path):
    from ultralytics import YOLO
    import torch
    model = YOLO(str(model_path))
    device = "0" if torch.cuda.is_available() else "cpu"
    res = model.val(
        data=str(data_yaml), imgsz=800, batch=4, device=device,
        verbose=False, plots=False,
    )
    return {
        "mAP50": float(res.box.map50),
        "mAP50-95": float(res.box.map),
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
    }


def evaluate_classifier(model_path: Path, data_dir: Path):
    from ultralytics import YOLO
    import torch
    model = YOLO(str(model_path))
    device = "0" if torch.cuda.is_available() else "cpu"
    res = model.val(
        data=str(data_dir), imgsz=384, batch=32, device=device,
        verbose=False, plots=False,
    )
    return {"top1": float(res.top1), "top5": float(res.top5)}


def detector_conf_sweep(model_path: Path, data_yaml: Path, thresholds=(0.1, 0.2, 0.3, 0.5)):
    """Real metric: how detector recall/precision trade off vs confidence.
    Uses the validation set. Returns list of dicts."""
    from ultralytics import YOLO
    import torch
    model = YOLO(str(model_path))
    device = "0" if torch.cuda.is_available() else "cpu"
    rows = []
    for t in thresholds:
        res = model.val(
            data=str(data_yaml), imgsz=800, batch=4, device=device,
            conf=t, verbose=False, plots=False,
        )
        rows.append({
            "conf_threshold": t,
            "precision": round(float(res.box.mp), 4),
            "recall": round(float(res.box.mr), 4),
            "mAP50": round(float(res.box.map50), 4),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect-model", default=str(DETECT_MODEL))
    ap.add_argument("--cls-model", default=str(CLS_MODEL))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "ablation"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detect_model = Path(args.detect_model)
    if not detect_model.exists():
        if OLD_DETECT.exists():
            print(f"⚠️  V2 detect model missing; using V1: {OLD_DETECT}")
            detect_model = OLD_DETECT
        else:
            print("❌ No detection model found. Train first.")
            detect_model = None

    cls_model = Path(args.cls_model)
    if not cls_model.exists():
        print("⚠️  Classifier model missing; skipping classifier metrics.")
        cls_model = None

    # ---- Real, runnable metrics ----
    det = None
    if detect_model:
        print(f"Evaluating detector: {detect_model}")
        det = evaluate_detector(detect_model, DATA_YAML)

    cls = None
    if cls_model:
        print(f"Evaluating classifier: {cls_model}")
        cls = evaluate_classifier(cls_model, CLS_DATA)

    # Composite "Vision system" metric: detector mAP50 weighted by downstream
    # classifier accuracy (detection must succeed AND fine-classify correctly).
    if det and cls:
        vision_overall = det["mAP50"] * cls["top1"]
    else:
        vision_overall = None

    # ---- Ablation table (one row per configuration) ----
    header = [
        "configuration", "detector_mAP50", "detector_mAP50-95",
        "classifier_top1", "composite_overall",
        "nir_accuracy", "mir_accuracy", "notes",
    ]
    rows = []

    rows.append({
        "configuration": "Vision-only",
        "detector_mAP50": round(det["mAP50"], 4) if det else MISSING,
        "detector_mAP50-95": round(det["mAP50-95"], 4) if det else MISSING,
        "classifier_top1": round(cls["top1"], 4) if cls else MISSING,
        "composite_overall": round(vision_overall, 4) if vision_overall is not None else MISSING,
        "nir_accuracy": MISSING,
        "mir_accuracy": MISSING,
        "notes": "Runnable. NIR/MIR cells require joint RGB+spectral dataset + trained SVM/RF.",
    })
    # NIR-only / MIR-only cannot be executed without hardware + spectral models.
    rows.append({
        "configuration": "NIR-only", "detector_mAP50": MISSING, "detector_mAP50-95": MISSING,
        "classifier_top1": MISSING, "composite_overall": MISSING,
        "nir_accuracy": MISSING, "mir_accuracy": MISSING,
        "notes": "REQUIRES NIR sensor + SVM; hardware not connected in this environment.",
    })
    rows.append({
        "configuration": "MIR-only", "detector_mAP50": MISSING, "detector_mAP50-95": MISSING,
        "classifier_top1": MISSING, "composite_overall": MISSING,
        "nir_accuracy": MISSING, "mir_accuracy": MISSING,
        "notes": "REQUIRES MIR/FTIR sensor + RF; hardware not connected in this environment.",
    })
    rows.append({
        "configuration": "Proposed (Vision + NIR->MIR gate)",
        "detector_mAP50": round(det["mAP50"], 4) if det else MISSING,
        "detector_mAP50-95": round(det["mAP50-95"], 4) if det else MISSING,
        "classifier_top1": round(cls["top1"], 4) if cls else MISSING,
        "composite_overall": round(vision_overall, 4) if vision_overall is not None else MISSING,
        "nir_accuracy": MISSING, "mir_accuracy": MISSING,
        "notes": "Vision components measured; spectral escalation requires hardware (see NIR/MIR rows).",
    })

    csv_path = out_dir / "modality_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"\n💾 Ablation CSV: {csv_path}")
    print("   (NIR/MIR cells = MISSING_HARDWARE until physical sensors + models exist)")

    # ---- Real detector confidence sweep ----
    if detect_model:
        sweep = detector_conf_sweep(detect_model, DATA_YAML)
        sweep_path = out_dir / "detector_conf_sweep.csv"
        with open(sweep_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["conf_threshold", "precision", "recall", "mAP50"])
            w.writeheader()
            w.writerows(sweep)
        print(f"💾 Detector confidence sweep: {sweep_path}")

        # Persist full metrics for the paper regeneration step
        metrics = {
            "detector": det,
            "classifier": cls,
            "vision_composite_overall": vision_overall,
            "detector_conf_sweep": sweep,
        }
        with open(out_dir / "ablation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Console summary
        print("\n" + "=" * 60)
        print("📊 ABLATION SUMMARY (measured components only)")
        print("=" * 60)
        print(f"  Detector mAP50   : {det['mAP50']*100:.1f}%  (mAP50-95 {det['mAP50-95']*100:.1f}%)")
        if cls:
            print(f"  Classifier top1  : {cls['top1']*100:.1f}%")
        if vision_overall is not None:
            print(f"  Vision composite : {vision_overall*100:.1f}%  (mAP50 x top1)")
        print("  NIR-only / MIR-only / spectral Proposed: not measurable here")
        print("=" * 60)


if __name__ == "__main__":
    main()
