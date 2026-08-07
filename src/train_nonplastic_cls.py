"""
train_nonplastic_cls.py — ML-based non-plastic classifier
=========================================================
Replaces the rule-based HSV + weight sorting logic used at the robotic-arm
stage with a lightweight YOLOv11n-cls model. It classifies a cropped,
detected *non-plastic* object into one of five bins:

    Paper, Metal, Glass, Organic, Other

The upstream detection model (YOLOv11) already separates PLASTIC from the
rest, so this branch only ever sees non-plastic crops. PLASTIC images are
deliberately excluded from training to keep the semantics of this classifier
clean (spectral sensing handles plastics downstream).

Source data: dataset_cls (10 classes). Mapping to the 5 target bins:
    paper       -> Paper
    metal       -> Metal
    glass       -> Glass
    biological  -> Organic
    cardboard   -> Other
    clothes     -> Other
    shoes       -> Other
    battery     -> Other
    trash       -> Other
    plastic     -> (excluded — handled by spectral branch)

Usage:
    python src/train_nonplastic_cls.py                  # build dataset + train
    python src/train_nonplastic_cls.py --epochs 60
    python src/train_nonplastic_cls.py --build-only     # just build the mapped dataset
    python src/train_nonplastic_cls.py --no-build       # assume dataset already built
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent

# dataset_cls class -> non-plastic target bin (None => excluded)
CLASS_MAP = {
    "paper": "Paper",
    "metal": "Metal",
    "glass": "Glass",
    "biological": "Organic",
    "cardboard": "Other",
    "clothes": "Other",
    "shoes": "Other",
    "battery": "Other",
    "trash": "Other",
    # "plastic": excluded on purpose
}

# Fixed, explicit target classes (order defines label ids)
TARGET_CLASSES = ["Paper", "Metal", "Glass", "Organic", "Other"]


def build_dataset(src_root: Path, dst_root: Path, force: bool = False):
    """Copy dataset_cls images into the 5-bin non-plastic layout."""
    print("\n" + "=" * 60)
    print("🗂️  BUILDING NON-PLASTIC DATASET")
    print("=" * 60)

    if dst_root.exists() and any(dst_root.iterdir()) and not force:
        print(f"  ⚠️  Target exists: {dst_root}")
        print(f"  💡 Use --force to rebuild. Skipping build.")
        return

    if dst_root.exists() and force:
        shutil.rmtree(dst_root)

    counts = {c: 0 for c in TARGET_CLASSES}
    for split in ("train", "val"):
        src_split = src_root / split
        if not src_split.exists():
            print(f"  ⚠️  Missing split: {src_split}")
            continue
        for src_class in src_split.iterdir():
            if not src_class.is_dir():
                continue
            target = CLASS_MAP.get(src_class.name)
            if target is None:
                # plastic (or unknown) — excluded from this branch
                continue
            dst_class = dst_root / split / target
            dst_class.mkdir(parents=True, exist_ok=True)
            for img in src_class.iterdir():
                if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    shutil.copy2(img, dst_class / img.name)
                    counts[target] += 1

    print("  ✅ Per-class image counts (train+val):")
    for c in TARGET_CLASSES:
        print(f"     {c:10s}: {counts[c]}")
    total = sum(counts.values())
    print(f"     {'TOTAL':10s}: {total}")
    if total == 0:
        print("  ❌ No images copied — check dataset_cls path.")
        sys.exit(1)


def train(args):
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  ✅ GPU: {gpu} ({vram:.1f} GB VRAM)")
        device = "0"
    else:
        print("  ⚠️  No GPU — training on CPU (slow)")
        device = "cpu"

    model = YOLO(args.model)
    print(f"  🏗️  Model: {args.model} (YOLOv11n-cls)")
    print(f"  📂 Data:  {args.data}")
    print(f"  🔄 Epochs: {args.epochs} | imgsz {args.imgsz} | batch {args.batch}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "nonplastic_cls"),
        name="train",
        exist_ok=True,
        save=True,
        save_period=10,
        plots=True,          # emits confusion_matrix.png automatically
        verbose=True,
        workers=2,
        optimizer="auto",
        lr0=0.01,
        pretrained=True,
        # Light augmentation — small model, plenty of real data
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5,
        fliplr=0.5, erasing=0.1,
    )
    return results


def evaluate_and_report(results, out_dir: Path):
    """Print metrics and dump a confusion-matrix CSV."""
    print("\n" + "=" * 60)
    print("📊 NON-PLASTIC CLASSIFIER REPORT")
    print("=" * 60)
    top1 = getattr(results, "top1", None)
    top5 = getattr(results, "top5", None)
    if top1 is not None:
        print(f"  Top-1 Accuracy: {top1*100:.1f}%")
    if top5 is not None:
        print(f"  Top-5 Accuracy: {top5*100:.1f}%")

    # Save confusion matrix to CSV when available
    cm = getattr(results, "confusion_matrix", None)
    if cm is not None and hasattr(cm, "matrix") and hasattr(cm, "names"):
        matrix = cm.matrix
        names = [str(n) for n in cm.names]
        csv_path = out_dir / "nonplastic_confusion_matrix.csv"
        with open(csv_path, "w", newline="") as f:
            f.write("true\\pred," + ",".join(names) + "\n")
            for i, row in enumerate(matrix):
                f.write(names[i] + "," + ",".join(f"{v:.4f}" for v in row) + "\n")
        print(f"  💾 Confusion matrix CSV: {csv_path}")
    else:
        print("  ⚠️  Built-in confusion matrix not exposed; see confusion_matrix.png in run dir.")


def main():
    parser = argparse.ArgumentParser(description="Train non-plastic YOLOv11n-cls classifier")
    parser.add_argument("--data", type=str,
                        default=str(PROJECT_ROOT / "dataset_nonplastic"),
                        help="Mapped 5-bin dataset dir (will be built from dataset_cls)")
    parser.add_argument("--src", type=str,
                        default=str(PROJECT_ROOT / "dataset_cls"),
                        help="Source 10-class dataset (dataset_cls)")
    parser.add_argument("--model", type=str, default="yolo11n-cls.pt",
                        help="Classification model (default: yolo11n-cls.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=224,
                        help="YOLOv11n-cls default input size")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--build-only", action="store_true",
                        help="Only build the mapped dataset, then exit")
    parser.add_argument("--no-build", action="store_true",
                        help="Skip dataset build (assume already built)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild mapped dataset even if it exists")
    args = parser.parse_args()

    if not args.no_build:
        build_dataset(Path(args.src), Path(args.data), force=args.force)
        if args.build_only:
            print("\n✅ Dataset build complete. Exiting (--build-only).")
            return

    print("\n" + "=" * 60)
    print("🗑️  NON-PLASTIC CLASSIFIER TRAINING")
    print("=" * 60)
    results = train(args)
    out_dir = PROJECT_ROOT / "runs" / "nonplastic_cls" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate_and_report(results, out_dir)

    print("\n" + "=" * 60)
    print("✅ DONE")
    print(f"  Best: runs/nonplastic_cls/train/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
