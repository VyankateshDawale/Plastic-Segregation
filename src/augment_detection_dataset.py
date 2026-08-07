"""
Offline YOLO detection dataset augmentation using photometric-only transforms.
Photometric transforms preserve bounding box coordinates, so label files are
copied as-is with an _aug suffix.

Usage:
    python src/augment_detection_dataset.py            # full run
    python src/augment_detection_dataset.py --dry-run   # process only 10 images
"""

import argparse
import os
import shutil
import random
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Photometric augmentation functions
# ---------------------------------------------------------------------------

def adjust_brightness(img, delta_range=0.30):
    """Randomly shift brightness by up to ±delta_range (fraction of 255)."""
    delta = random.uniform(-delta_range, delta_range) * 255.0
    return np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def adjust_contrast(img, low=0.7, high=1.3):
    """Randomly scale contrast around the image mean."""
    factor = random.uniform(low, high)
    mean = img.mean()
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def add_gaussian_noise(img, sigma_low=5, sigma_high=15):
    """Add Gaussian noise with a random sigma."""
    sigma = random.uniform(sigma_low, sigma_high)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_motion_blur(img, kmin=3, kmax=7):
    """Horizontal motion blur to simulate conveyor belt movement."""
    k = random.choice(range(kmin, kmax + 1, 2))  # odd kernel sizes
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def color_jitter(img, hue_delta=10, sat_range=0.30):
    """Random hue shift (±hue_delta) and saturation scale (±sat_range)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Hue: channel 0 is [0..180] in OpenCV
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-hue_delta, hue_delta)) % 180.0
    # Saturation
    sat_factor = random.uniform(1.0 - sat_range, 1.0 + sat_range)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def gamma_correction(img, low=0.7, high=1.5):
    """Random gamma correction."""
    gamma = random.uniform(low, high)
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(img, table)


# All available transforms
TRANSFORMS = [
    adjust_brightness,
    adjust_contrast,
    add_gaussian_noise,
    apply_motion_blur,
    color_jitter,
    gamma_correction,
]


def augment_image(img):
    """Apply a random subset (2-4) of photometric transforms in random order."""
    n = random.randint(2, 4)
    chosen = random.sample(TRANSFORMS, n)
    for fn in chosen:
        img = fn(img)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Augment YOLO detection dataset (photometric only)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only 10 images for a quick test",
    )
    args = parser.parse_args()

    base = r"D:\projects\Plastic Waste Sorting System\dataset\GARBAGE CLASSIFICATION\train"
    img_dir = os.path.join(base, "images")
    lbl_dir = os.path.join(base, "labels")

    # Gather all .jpg images (exclude any already-augmented ones)
    all_imgs = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith(".jpg") and "_aug" not in f
    )

    if args.dry_run:
        all_imgs = all_imgs[:10]
        print(f"[DRY-RUN] Processing {len(all_imgs)} images only.\n")
    else:
        print(f"Found {len(all_imgs)} original images to augment.\n")

    t0 = time.time()
    created = 0
    skipped = 0

    for idx, fname in enumerate(all_imgs, 1):
        stem, ext = os.path.splitext(fname)
        aug_img_name = f"{stem}_aug{ext}"
        aug_lbl_name = f"{stem}_aug.txt"

        aug_img_path = os.path.join(img_dir, aug_img_name)
        aug_lbl_path = os.path.join(lbl_dir, aug_lbl_name)

        # Skip if augmented file already exists
        if os.path.exists(aug_img_path):
            skipped += 1
            continue

        # Read original image
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            print(f"  [WARN] Could not read {fname}, skipping.")
            skipped += 1
            continue

        # Apply photometric augmentation
        aug = augment_image(img)

        # Save augmented image
        cv2.imwrite(aug_img_path, aug)

        # Copy label file (identical content — bboxes unchanged)
        src_lbl = os.path.join(lbl_dir, f"{stem}.txt")
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, aug_lbl_path)
        else:
            # Some images may lack labels (background / negatives) — that's OK
            pass

        created += 1

        if idx % 500 == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            print(f"  [{idx}/{len(all_imgs)}] created so far: {created} | "
                  f"{rate:.1f} img/s | elapsed {elapsed:.1f}s")

    elapsed = time.time() - t0
    total_images = len([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])

    print(f"\nDone in {elapsed:.1f}s.")
    print(f"  Augmented images created : {created}")
    print(f"  Skipped (existing/error) : {skipped}")
    print(f"  Total images in folder   : {total_images}")


if __name__ == "__main__":
    main()
