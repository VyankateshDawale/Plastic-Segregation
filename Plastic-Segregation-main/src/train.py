"""
train.py — Train YOLOv11 on the Garbage Detection Dataset
==========================================================

Usage:
    python src/train.py                     # Train with default settings
    python src/train.py --epochs 150        # Custom epochs
    python src/train.py --resume            # Resume interrupted training
    python src/train.py --model yolo11n.pt  # Use nano model instead
"""

import argparse
import sys
import os
import torch
from pathlib import Path
from ultralytics import YOLO


def check_gpu():
    """Check GPU availability and print info."""
    print("=" * 60)
    print("🔍 GPU CHECK")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  ✅ GPU Found: {gpu_name}")
        print(f"  ✅ VRAM: {gpu_memory:.1f} GB")
        print(f"  ✅ CUDA Version: {torch.version.cuda}")
        print(f"  ✅ PyTorch CUDA: {torch.cuda.is_available()}")
        return "0"  # GPU device index
    else:
        print("  ⚠️  No GPU detected. Training on CPU (will be slow).")
        print("  💡 Tip: Install PyTorch with CUDA support:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        return "cpu"


def find_data_yaml():
    """Find the data.yaml file in the dataset directory."""
    project_root = Path(__file__).parent.parent
    possible_paths = [
        project_root / "dataset" / "data.yaml",
        project_root / "dataset" / "GARBAGE CLASSIFICATION" / "data.yaml",
        project_root / "GARBAGE CLASSIFICATION" / "data.yaml",
        project_root / "data.yaml",
    ]

    for path in possible_paths:
        if path.exists():
            print(f"  ✅ Found data.yaml: {path}")
            return str(path)

    # Search recursively
    for yaml_file in project_root.rglob("data.yaml"):
        print(f"  ✅ Found data.yaml: {yaml_file}")
        return str(yaml_file)

    print("  ❌ data.yaml not found!")
    print("  📂 Please place the downloaded dataset in one of:")
    for p in possible_paths:
        print(f"      {p.parent}")
    sys.exit(1)


def fix_data_yaml_paths(data_yaml_path):
    """
    Fix data.yaml paths to be absolute paths pointing to the actual dataset.
    Many Kaggle datasets have relative paths that break on different machines.
    """
    import yaml

    data_yaml = Path(data_yaml_path)
    dataset_dir = data_yaml.parent

    with open(data_yaml, "r") as f:
        config = yaml.safe_load(f)

    # Check if paths need fixing
    needs_fix = False

    for split in ["train", "val", "valid", "test"]:
        if split in config:
            split_path = Path(config[split])
            if not split_path.is_absolute():
                # Make it absolute relative to dataset dir
                abs_path = dataset_dir / split_path
                if not abs_path.exists():
                    # Try common alternatives
                    alternatives = [
                        dataset_dir / split / "images",
                        dataset_dir / split,
                    ]
                    for alt in alternatives:
                        if alt.exists():
                            abs_path = alt
                            break

                config[split] = str(abs_path)
                needs_fix = True

    # Normalize val/valid naming
    if "valid" in config and "val" not in config:
        config["val"] = config.pop("valid")
        needs_fix = True

    if needs_fix:
        # Backup original
        backup_path = data_yaml.with_suffix(".yaml.bak")
        if not backup_path.exists():
            import shutil
            shutil.copy2(data_yaml, backup_path)
            print(f"  📋 Backed up original data.yaml to {backup_path.name}")

        with open(data_yaml, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print("  🔧 Fixed data.yaml paths to absolute paths")

    return str(data_yaml)


def train(args):
    """Main training function."""
    print()
    print("🗑️  PLASTIC WASTE SORTING SYSTEM — MODEL TRAINING")
    print("=" * 60)

    # Step 1: Check GPU
    device = check_gpu()
    print()

    # Step 2: Find dataset
    print("=" * 60)
    print("📂 DATASET DISCOVERY")
    print("=" * 60)
    data_yaml = args.data if args.data else find_data_yaml()
    data_yaml = fix_data_yaml_paths(data_yaml)
    print()

    # Step 3: Load model
    print("=" * 60)
    print("🤖 MODEL SETUP")
    print("=" * 60)

    if args.resume:
        # Resume from last checkpoint
        last_pt = Path("runs/detect/train/weights/last.pt")
        if last_pt.exists():
            model = YOLO(str(last_pt))
            print(f"  ✅ Resuming from: {last_pt}")
        else:
            print("  ❌ No checkpoint found to resume from.")
            print("  Starting fresh training instead...")
            model = YOLO(args.model)
            print(f"  ✅ Loaded pretrained: {args.model}")
    else:
        model = YOLO(args.model)
        print(f"  ✅ Loaded pretrained: {args.model}")
    print()

    # Step 4: Train
    print("=" * 60)
    print("🚀 STARTING TRAINING")
    print("=" * 60)
    print(f"  📊 Epochs: {args.epochs}")
    print(f"  📐 Image Size: {args.imgsz}")
    print(f"  📦 Batch Size: {args.batch}")
    print(f"  🎯 Device: {'GPU (RTX 4050)' if device == '0' else 'CPU'}")
    print(f"  📁 Project: {args.project}")
    print()

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save=True,
        save_period=10,         # Save checkpoint every 10 epochs
        plots=True,             # Generate training plots
        verbose=True,
        workers=2,              # DataLoader workers (lower = less RAM)
        exist_ok=True,          # Overwrite existing runs
        pretrained=True,
        optimizer="auto",       # AdamW by default
        lr0=0.01,               # Initial learning rate
        lrf=0.01,               # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        # Augmentation
        hsv_h=0.015,            # HSV-Hue augmentation
        hsv_s=0.7,              # HSV-Saturation
        hsv_v=0.4,              # HSV-Value
        degrees=10.0,           # Rotation
        translate=0.1,          # Translation
        scale=0.5,              # Scale
        fliplr=0.5,             # Horizontal flip
        flipud=0.0,             # No vertical flip
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.0,              # MixUp disabled (saves RAM)
    )

    print()
    print("=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  📁 Results saved to: {args.project}/{args.name}/")
    print(f"  🏆 Best weights: {args.project}/{args.name}/weights/best.pt")
    print()
    print("  Next steps:")
    print("    1. Check training plots in the results folder")
    print("    2. Run validation: python src/validate.py")
    print("    3. Start detection: python src/detect.py")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 on Garbage Detection Dataset"
    )
    parser.add_argument(
        "--model", type=str, default="yolo11s.pt",
        help="Pretrained model to start from (default: yolo11s.pt)"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to data.yaml (auto-detected if not specified)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="Batch size (default: 8, increase to 16 if enough RAM)"
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Early stopping patience (default: 20)"
    )
    parser.add_argument(
        "--project", type=str, default="runs/detect",
        help="Project directory for saving results"
    )
    parser.add_argument(
        "--name", type=str, default="train",
        help="Run name (default: train)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
