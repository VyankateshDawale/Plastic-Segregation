"""
Train YOLOv11 Classification Model for Waste Sorting.
Uses YOLO classify mode for image classification (10 classes).
"""
import argparse
import torch
from ultralytics import YOLO


def train(args):
    print("=" * 60)
    print("  🗑️  Waste Classification Training (YOLOv11)")
    print("=" * 60)
    
    # GPU check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  ✅ GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
    else:
        print("  ⚠️  No GPU — training on CPU (slow)")
    
    print(f"  📂 Dataset: {args.data}")
    print(f"  🏗️  Model: {args.model}")
    print(f"  📐 Image Size: {args.imgsz}")
    print(f"  📦 Batch Size: {args.batch}")
    print(f"  🔄 Epochs: {args.epochs}")
    print("=" * 60)
    
    # Load model
    if args.resume:
        # Resume from last checkpoint
        last_pt = r"D:\projects\Plastic Waste Sorting System\runs\classify\train\weights\last.pt"
        print(f"\n  🔄 Resuming from: {last_pt}")
        model = YOLO(last_pt)
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            device=0 if torch.cuda.is_available() else "cpu",
            
            # Project settings
            project=r"D:\projects\Plastic Waste Sorting System\runs\classify",
            name="train",
            exist_ok=True,
            
            # Save settings
            save=True,
            save_period=10,
            plots=True,
            verbose=True,
            
            # Optimization
            workers=2,
            optimizer="auto",
            lr0=0.01,
            
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            erasing=0.1,
            
            # Pretrained
            pretrained=True,
        )
    
    print("\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n  📊 Results saved to: runs/classify/train/")
    print(f"  🏆 Best model: runs/classify/train/weights/best.pt")
    print(f"  📦 Last model: runs/classify/train/weights/last.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Waste Classification Model")
    parser.add_argument(
        "--data", type=str,
        default=r"D:\projects\Plastic Waste Sorting System\dataset_cls",
        help="Path to classification dataset"
    )
    parser.add_argument(
        "--model", type=str, default="yolo11s-cls.pt",
        help="YOLO classification model (default: yolo11s-cls.pt)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=384,
        help="Input image size (default: 384)"
    )
    parser.add_argument(
        "--batch", type=int, default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Early stopping patience (default: 20)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
