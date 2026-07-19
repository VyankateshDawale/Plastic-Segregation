"""
pipeline_multimodal.py — Phase 1 integration: YOLO vision → NIR/MIR fallback

Aligns repo runtime with the paper architecture at a software level:
  1) Run YOLOv11 on an image / webcam frame
  2) If top class is PLASTIC (and optional NIR/MIR vectors are provided),
     run Atharva_NIR DualStagePlasticInference
  3) Print a combined routing decision

This does NOT require live Specim/FTIR hardware. Pass precomputed spectra
as .npy vectors when available.

Usage:
  python src/pipeline_multimodal.py --image sample.jpg --weights models/best.pt
  python src/pipeline_multimodal.py --image sample.jpg --weights models/best.pt \\
      --nir nir_232.npy --mir mir_3736.npy --models-dir models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Atharva_NIR"))

# Reuse bin map from detect.py when available
try:
    from detect import SORTING_MAP, find_best_weights
except Exception:  # pragma: no cover
    SORTING_MAP = {}
    find_best_weights = lambda: None  # noqa: E731


RESIN_BIN_MAP = {
    "PET": ("BLUE BIN (DRY RECYCLABLE)", "Resin #1 — rinse and recycle"),
    "HDPE": ("BLUE BIN (DRY RECYCLABLE)", "Resin #2 — rinse and recycle"),
    "PVC": ("SPECIAL COLLECTION", "Resin #3 — do not mix with dry recyclables"),
    "LDPE": ("BLUE BIN (DRY RECYCLABLE)", "Resin #4 — check local film rules"),
    "PP": ("BLUE BIN (DRY RECYCLABLE)", "Resin #5 — rinse and recycle"),
    "PS": ("CHECK LOCAL RULES", "Resin #6 — often limited acceptance"),
}


def load_yolo(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def vision_predict(model, image_path: str, conf: float = 0.35):
    results = model.predict(source=image_path, conf=conf, verbose=False)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append(
                {
                    "class": str(names[cls_id]).upper(),
                    "confidence": float(box.conf[0]),
                }
            )
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def spectral_predict(models_dir: str, nir_path: str | None, mir_path: str | None):
    from inference.inference import DualStagePlasticInference

    clf = DualStagePlasticInference(models_dir=models_dir)
    raw_nir = np.load(nir_path) if nir_path else None
    raw_mir = np.load(mir_path) if mir_path else None
    if raw_nir is None and raw_mir is None:
        raise ValueError("Provide --nir and/or --mir .npy spectra for spectral stage")
    return clf.classify_sample(raw_nir=raw_nir, raw_mir=raw_mir)


def route_vision(cls_name: str) -> str:
    info = SORTING_MAP.get(cls_name, {})
    return info.get("bin", "UNKNOWN BIN") + " — " + info.get("instruction", "manual sort")


def route_resin(resin: str) -> str:
    bin_name, detail = RESIN_BIN_MAP.get(resin, ("MANUAL SORT", "Unknown resin"))
    return f"{bin_name} — {detail}"


def main():
    parser = argparse.ArgumentParser(description="YOLO → NIR/MIR multimodal handoff")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--weights", default=None, help="YOLO .pt weights")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--models-dir", default=str(ROOT / "models"))
    parser.add_argument("--nir", default=None, help="Optional NIR vector .npy (232,)")
    parser.add_argument("--mir", default=None, help="Optional MIR vector .npy (3736,)")
    args = parser.parse_args()

    weights = args.weights or find_best_weights()
    if weights is None:
        print("ERROR: No YOLO weights found. Place models/best.pt or pass --weights")
        sys.exit(1)
    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    print("=" * 60)
    print("MULTIMODAL PIPELINE — Phase 1 (YOLO → NIR/MIR handoff)")
    print("=" * 60)
    print(f"Weights: {weights}")
    print(f"Image:   {args.image}")

    model = load_yolo(weights)
    detections = vision_predict(model, args.image, conf=args.conf)

    if not detections:
        print("Vision: no detections above confidence threshold")
        sys.exit(0)

    top = detections[0]
    print(f"Vision top: {top['class']} ({top['confidence']:.1%})")
    for d in detections[:5]:
        print(f"  - {d['class']}: {d['confidence']:.1%}")

    if top["class"] != "PLASTIC":
        print(f"Routing (vision): {route_vision(top['class'])}")
        print("Spectral stage skipped (item not classified as Plastic).")
        return

    print("Vision class is PLASTIC → invoking spectral stage")
    if not args.nir and not args.mir:
        print(
            "NOTE: No --nir/--mir provided. Spectral stage not run.\n"
            "      Export spectra as .npy and re-run to complete paper-aligned flow."
        )
        print(f"Routing (vision-only plastic): {route_vision('PLASTIC')}")
        return

    try:
        spectral = spectral_predict(args.models_dir, args.nir, args.mir)
    except FileNotFoundError as exc:
        print(f"ERROR: Spectral models missing under {args.models_dir}: {exc}")
        print("Place NIR/MIR joblib pickles in models/ (see models/README.md)")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Spectral inference failed: {exc}")
        sys.exit(1)

    resin = str(spectral.get("predicted_class", "UNKNOWN"))
    print("--- SPECTRAL RESULT ---")
    print(f"Sensor used: {spectral.get('sensor_used')}")
    print(f"Predicted:   {resin} ({float(spectral.get('confidence', 0)) * 100:.2f}%)")
    print(f"Fallback:    {spectral.get('fallback_triggered')}")
    if spectral.get("fallback_reason"):
        print(f"Reason:      {spectral.get('fallback_reason')}")
    print(f"Routing (resin): {route_resin(resin)}")


if __name__ == "__main__":
    main()
