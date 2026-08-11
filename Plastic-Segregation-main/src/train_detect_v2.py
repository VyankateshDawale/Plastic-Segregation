"""
train_detect_v2.py — Upgraded YOLOv11m Detection Training
==========================================================
Key improvements over v1 (train.py):
  - YOLOv11m model (vs yolo11s) — 2x more parameters, much higher capacity
  - Image size 800 (vs 640) — better for small waste objects
  - Aggressive augmentation: mosaic, mixup, copy_paste
  - close_mosaic=20 for clean fine-tuning in final epochs
  - Reports mAP@50 AND mAP@50-95
  - 200 epochs with patience=30
  
Usage:
    python src/train_detect_v2.py                        # Full training
    python src/train_detect_v2.py --model yolo11l.pt     # Use large model
    python src/train_detect_v2.py --resume               # Resume training
    python src/train_detect_v2.py --batch 4 --imgsz 640  # Memory-safe mode
"""

import argparse
import sys
import os
import torch
from pathlib import Path
from ultralytics import YOLO


def check_gpu():
    """Check GPU availability and return device string."""
    print("=" * 60)
    print("🔍 GPU CHECK")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3) if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  ✅ GPU: {gpu_name}")
        print(f"  ✅ VRAM: {gpu_mem:.1f} GB")
        print(f"  ✅ CUDA: {torch.version.cuda}")
        
        # Estimate safe batch size based on VRAM
        if gpu_mem >= 8:
            suggested_batch = 8
        elif gpu_mem >= 6:
            suggested_batch = 4
        else:
            suggested_batch = 2
        print(f"  💡 Suggested batch size: {suggested_batch}")
        return "0", suggested_batch
    else:
        print("  ⚠️  No GPU detected. Training will be extremely slow.")
        return "cpu", 2


def find_data_yaml():
    """Find the data.yaml file in the dataset directory."""
    project_root = Path(__file__).parent.parent
    candidates = [
        project_root / "dataset" / "GARBAGE CLASSIFICATION" / "data.yaml",
        project_root / "dataset" / "data.yaml",
        project_root / "data.yaml",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    
    # Recursive fallback
    for yaml_file in project_root.rglob("data.yaml"):
        if "venv" not in str(yaml_file):
            return str(yaml_file)
    
    print("❌ data.yaml not found!")
    sys.exit(1)


def train(args):
    """Main training function with upgraded config."""
    print()
    print("🗑️  WASTE DETECTION V2 — UPGRADED TRAINING")
    print("=" * 60)
    print(f"  🏗️  Model:      {args.model} (upgraded from yolo11s)")
    print(f"  📐 Image Size:  {args.imgsz} (upgraded from 640)")
    print(f"  🔄 Epochs:      {args.epochs}")
    print(f"  📦 Batch Size:  {args.batch}")
    print(f"  ⏳ Patience:    {args.patience}")
    print("=" * 60)
    
    # GPU check
    device, suggested_batch = check_gpu()
    if args.batch > suggested_batch and device != "cpu":
        print(f"\n  ⚠️  Batch {args.batch} may OOM on {torch.cuda.get_device_name(0)}.")
        print(f"  💡 Using suggested batch: {suggested_batch}")
        args.batch = suggested_batch
    print()

    # Find dataset
    data_yaml = args.data if args.data else find_data_yaml()
    print(f"  📂 Dataset: {data_yaml}")
    print()

    # Load model
    if args.resume:
        last_pt = Path(args.project) / args.name / "weights" / "last.pt"
        if last_pt.exists():
            model = YOLO(str(last_pt))
            print(f"  🔄 Resuming from: {last_pt}")
        else:
            print(f"  ❌ No checkpoint at {last_pt}")
            print(f"  Starting fresh with {args.model}")
            model = YOLO(args.model)
    else:
        model = YOLO(args.model)
        print(f"  ✅ Loaded: {args.model}")
    
    print()
    print("🚀 TRAINING CONFIG (V2 UPGRADES)")
    print("-" * 40)
    print("  ✨ Augmentation: mosaic=1.0, mixup=0.15, copy_paste=0.1")
    print("  ✨ close_mosaic=20 (clean final epochs)")
    print("  ✨ degrees=15, scale=0.7, translate=0.15")
    print("  ✨ erasing=0.2 (random erasing)")
    print("  ✨ amp=True (mixed precision)")
    print()

    # ── TRAINING ──
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        
        # Save & logging
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        
        # Optimization
        workers=2,
        patience=args.patience,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,              # Lower LR for medium/large models
        lrf=0.01,               # Final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,        # Longer warmup for larger model
        warmup_momentum=0.8,
        
        # Mixed precision
        amp=True,               # FP16 training — saves VRAM
        
        # ── AGGRESSIVE AUGMENTATION (V2 key upgrade) ──
        hsv_h=0.02,             # HSV-Hue (slightly more than default)
        hsv_s=0.7,              # HSV-Saturation
        hsv_v=0.5,              # HSV-Value (brightness)
        degrees=15.0,           # Rotation ±15° (waste is randomly oriented)
        translate=0.15,         # Translation ±15%
        scale=0.7,              # Scale ±70% (waste varies in size)
        shear=5.0,              # Shear ±5°
        perspective=0.001,      # Slight perspective warp
        fliplr=0.5,             # Horizontal flip
        flipud=0.1,             # Slight vertical flip (conveyor can be any direction)
        mosaic=1.0,             # Mosaic augmentation (100%)
        mixup=0.15,             # MixUp — blend images together
        copy_paste=0.1,         # Copy-paste augmentation
        erasing=0.2,            # Random erasing (occlusion simulation)
        close_mosaic=20,        # Disable mosaic in last 20 epochs for clean fine-tuning
    )

    print()
    print("=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  📁 Results: {args.project}/{args.name}/")
    print(f"  🏆 Best:    {args.project}/{args.name}/weights/best.pt")
    print()
    
    # ── Run validation and print final metrics ──
    print("📊 FINAL VALIDATION")
    print("-" * 40)
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.exists():
        val_model = YOLO(str(best_pt))
        val_results = val_model.val(
            data=data_yaml,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            plots=True,
            save_json=True,
        )
        
        print(f"\n  📈 mAP@50:    {val_results.box.map50:.4f} ({val_results.box.map50*100:.1f}%)")
        print(f"  📈 mAP@50-95: {val_results.box.map:.4f} ({val_results.box.map*100:.1f}%)")
        print(f"  📈 Precision:  {val_results.box.mp:.4f}")
        print(f"  📈 Recall:     {val_results.box.mr:.4f}")
        
        # Per-class AP
        class_names = val_model.names
        print(f"\n  Per-Class AP@50:")
        for i, ap50 in enumerate(val_results.box.ap50):
            name = class_names.get(i, f"class_{i}")
            print(f"    {name:15s}: {ap50*100:.1f}%")
    
    print()
    return results


def main():
    parser = argparse.ArgumentParser(description="V2: Upgraded Detection Training")
    parser.add_argument("--model", type=str, default="yolo11m.pt",
                        help="Model (default: yolo11m.pt)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Epochs (default: 200)")
    parser.add_argument("--imgsz", type=int, default=800,
                        help="Image size (default: 800)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size (default: 4 for RTX 4050)")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (default: 30)")
    parser.add_argument("--project", type=str,
                        default=r"D:\projects\Plastic Waste Sorting System\runs\detect_v2",
                        help="Project directory")
    parser.add_argument("--name", type=str, default="train",
                        help="Run name")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
