"""
validate.py — Validate the trained model on the test set
=========================================================

Usage:
    python src/validate.py
    python src/validate.py --weights runs/detect/train/weights/best.pt
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


def find_best_weights():
    """Find the best trained model weights."""
    project_root = Path(__file__).parent.parent
    for pt_file in project_root.rglob("best.pt"):
        return str(pt_file)
    return None


def find_data_yaml():
    """Find the data.yaml file."""
    project_root = Path(__file__).parent.parent
    for yaml_file in project_root.rglob("data.yaml"):
        return str(yaml_file)
    return None


def main():
    parser = argparse.ArgumentParser(description="Validate trained model")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    print()
    print("🗑️  WASTE SORTING SYSTEM — MODEL VALIDATION")
    print("=" * 60)

    # Find weights
    weights = args.weights or find_best_weights()
    if weights is None:
        print("  ❌ No trained model found! Train first: python src/train.py")
        sys.exit(1)

    # Find data
    data = args.data or find_data_yaml()
    if data is None:
        print("  ❌ data.yaml not found!")
        sys.exit(1)

    print(f"  ✅ Weights: {weights}")
    print(f"  ✅ Data: {data}")
    print()

    # Load model and validate
    model = YOLO(weights)
    results = model.val(
        data=data,
        imgsz=args.imgsz,
        batch=args.batch,
        split="test",  # Validate on test set
        verbose=True,
        plots=True,
    )

    print()
    print("=" * 60)
    print("📊 VALIDATION RESULTS")
    print("=" * 60)
    print(f"  mAP@50:      {results.box.map50:.4f}")
    print(f"  mAP@50-95:   {results.box.map:.4f}")
    print()
    print("  Per-class AP@50:")
    for i, cls_name in enumerate(results.names.values()):
        ap = results.box.ap50[i] if i < len(results.box.ap50) else 0
        print(f"    {cls_name:15s}: {ap:.4f}")
    print()
    print("  📁 Plots saved to validation results directory")
    print()


if __name__ == "__main__":
    main()
