"""
sanity_check.py — Verify dataset integrity before training
===========================================================

Run this BEFORE training to catch any issues:
    python src/sanity_check.py

Checks:
    1. data.yaml exists and is valid
    2. All image/label directories exist
    3. Image-label pairing is correct
    4. Class distribution is balanced
    5. Sample annotations are visualized
"""

import sys
import os
import random
from pathlib import Path
from collections import Counter

import cv2
import yaml
import numpy as np


# Colors for each class
CLASS_COLORS = [
    (0, 200, 0),       # BIODEGRADABLE — Green
    (200, 150, 0),     # CARDBOARD — Blue
    (0, 200, 200),     # GLASS — Yellow
    (0, 165, 255),     # METAL — Orange
    (255, 150, 50),    # PAPER — Light blue
    (0, 100, 255),     # PLASTIC — Red
]


def find_data_yaml():
    """Find data.yaml in the project."""
    project_root = Path(__file__).parent.parent
    for yaml_file in project_root.rglob("data.yaml"):
        return yaml_file
    return None


def main():
    print()
    print("🔍 DATASET SANITY CHECK")
    print("=" * 60)

    # Step 1: Find and parse data.yaml
    data_yaml_path = find_data_yaml()
    if data_yaml_path is None:
        print("  ❌ data.yaml not found!")
        print("  📂 Place the dataset in the project directory.")
        sys.exit(1)

    print(f"  ✅ Found: {data_yaml_path}")

    with open(data_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"  📋 Classes ({config.get('nc', '?')}): {config.get('names', [])}")
    print()

    dataset_root = data_yaml_path.parent
    total_images = 0
    total_labels = 0
    all_class_counts = Counter()

    for split in ["train", "val", "valid", "test"]:
        if split not in config:
            continue

        split_path = Path(config[split])
        if not split_path.is_absolute():
            split_path = dataset_root / split_path

        # Figure out images and labels paths
        if split_path.name == "images":
            images_dir = split_path
            labels_dir = split_path.parent / "labels"
        else:
            images_dir = split_path / "images" if (split_path / "images").exists() else split_path
            labels_dir = split_path / "labels" if (split_path / "labels").exists() else split_path.parent / "labels"

        print(f"  📂 {split.upper()}")

        if not images_dir.exists():
            print(f"     ❌ Images directory not found: {images_dir}")
            continue

        # Count images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        n_images = len(image_files)
        total_images += n_images
        print(f"     📷 Images: {n_images}")

        if not labels_dir.exists():
            print(f"     ❌ Labels directory not found: {labels_dir}")
            continue

        # Count labels
        label_files = list(labels_dir.glob("*.txt"))
        n_labels = len(label_files)
        total_labels += n_labels
        print(f"     🏷️  Labels: {n_labels}")

        # Check pairing
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}
        missing_labels = image_stems - label_stems
        orphan_labels = label_stems - image_stems

        if missing_labels:
            print(f"     ⚠️  {len(missing_labels)} images missing labels")
        if orphan_labels:
            print(f"     ⚠️  {len(orphan_labels)} orphan labels (no matching image)")
        if not missing_labels and not orphan_labels:
            print(f"     ✅ All images have matching labels")

        # Count class distribution
        split_class_counts = Counter()
        for lbl in label_files:
            with open(lbl, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        split_class_counts[cls_id] += 1
                        all_class_counts[cls_id] += 1

        if split_class_counts:
            names = config.get("names", {})
            if isinstance(names, list):
                names = {i: n for i, n in enumerate(names)}
            print(f"     📊 Annotations: {sum(split_class_counts.values())}")
            for cls_id, count in sorted(split_class_counts.items()):
                cls_name = names.get(cls_id, f"class_{cls_id}")
                print(f"        {cls_name}: {count}")

        print()

    # Summary
    print("=" * 60)
    print("📊 OVERALL SUMMARY")
    print("=" * 60)
    print(f"  Total images: {total_images}")
    print(f"  Total labels: {total_labels}")
    print(f"  Total annotations: {sum(all_class_counts.values())}")
    print()

    # Visualize a random sample
    print("  🖼️  Showing 4 random annotated samples...")
    print("     (Close the window to continue)")
    print()

    # Find train images for visualization
    train_path = config.get("train", config.get("val", ""))
    train_path = Path(train_path)
    if not train_path.is_absolute():
        train_path = dataset_root / train_path

    if train_path.name == "images":
        images_dir = train_path
        labels_dir = train_path.parent / "labels"
    else:
        images_dir = train_path / "images" if (train_path / "images").exists() else train_path
        labels_dir = images_dir.parent / "labels"

    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        samples = random.sample(image_files, min(4, len(image_files)))
        names = config.get("names", {})
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}

        for img_path in samples:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            label_path = labels_dir / (img_path.stem + ".txt")

            if label_path.exists():
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])

                            # Convert YOLO format to pixel coords
                            x1 = int((cx - bw / 2) * w)
                            y1 = int((cy - bh / 2) * h)
                            x2 = int((cx + bw / 2) * w)
                            y2 = int((cy + bh / 2) * h)

                            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                            label = names.get(cls_id, f"class_{cls_id}")
                            cv2.putText(img, label, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Resize for display if too large
            max_dim = 800
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale)

            cv2.imshow(f"Sample: {img_path.name}", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("  ✅ Sanity check complete!")
    print()


if __name__ == "__main__":
    main()
