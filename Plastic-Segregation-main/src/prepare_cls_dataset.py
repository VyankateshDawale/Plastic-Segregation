"""
Prepare classification dataset by splitting into train/val sets.
YOLO classify expects: dataset_root/train/class_name/images and dataset_root/val/class_name/images
"""
import os
import shutil
import random

SOURCE_DIR = r"D:\projects\Plastic Waste Sorting System\standardized_384"
OUTPUT_DIR = r"D:\projects\Plastic Waste Sorting System\dataset_cls"
TRAIN_RATIO = 0.8
SEED = 42

def main():
    random.seed(SEED)
    
    classes = sorted([d for d in os.listdir(SOURCE_DIR) 
                      if os.path.isdir(os.path.join(SOURCE_DIR, d))])
    
    print(f"📦 Found {len(classes)} classes: {', '.join(classes)}")
    print(f"📂 Source: {SOURCE_DIR}")
    print(f"📂 Output: {OUTPUT_DIR}")
    print(f"📊 Split: {int(TRAIN_RATIO*100)}% train / {int((1-TRAIN_RATIO)*100)}% val")
    print()
    
    # Create output directories
    for split in ["train", "val"]:
        for cls in classes:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)
    
    total_train = 0
    total_val = 0
    
    for cls in classes:
        cls_dir = os.path.join(SOURCE_DIR, cls)
        images = [f for f in os.listdir(cls_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to train
        for img in train_images:
            src = os.path.join(cls_dir, img)
            dst = os.path.join(OUTPUT_DIR, "train", cls, img)
            shutil.copy2(src, dst)
        
        # Copy to val
        for img in val_images:
            src = os.path.join(cls_dir, img)
            dst = os.path.join(OUTPUT_DIR, "val", cls, img)
            shutil.copy2(src, dst)
        
        total_train += len(train_images)
        total_val += len(val_images)
        
        print(f"  ✅ {cls:15s} → train: {len(train_images):5d} | val: {len(val_images):4d}")
    
    print()
    print(f"📊 Total: {total_train + total_val} images")
    print(f"   Train: {total_train} images")
    print(f"   Val:   {total_val} images")
    print(f"\n✅ Dataset ready at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
