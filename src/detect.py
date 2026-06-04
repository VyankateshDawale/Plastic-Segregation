"""
detect.py — Real-Time Webcam Waste Detection & Sorting Recommendation
=====================================================================

Usage:
    python src/detect.py                                    # Default webcam
    python src/detect.py --source 1                         # External webcam
    python src/detect.py --weights runs/detect/train/weights/best.pt
    python src/detect.py --conf 0.5                         # Higher confidence threshold

Controls:
    Q / ESC     — Quit
    S           — Screenshot current frame
    P           — Pause/Resume
    +/-         — Increase/Decrease confidence threshold
    R           — Reset statistics
"""

import argparse
import cv2
import numpy as np
import time
import sys
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════
# SORTING CONFIGURATION
# ══════════════════════════════════════════════════════════════

# Class → Bin mapping with sorting instructions
SORTING_MAP = {
    "BIODEGRADABLE": {
        "bin": "🟢 GREEN BIN",
        "color": (0, 200, 0),           # Green
        "bg_color": (0, 80, 0),
        "instruction": "Compost / Organic Waste",
        "details": "Food scraps, yard waste, paper towels",
        "icon": "🌿",
    },
    "CARDBOARD": {
        "bin": "🔵 BLUE BIN",
        "color": (200, 150, 0),          # Blue-ish
        "bg_color": (80, 50, 0),
        "instruction": "Recyclable — Flatten before disposal",
        "details": "Boxes, packaging, corrugated cardboard",
        "icon": "📦",
    },
    "GLASS": {
        "bin": "🟡 YELLOW BIN",
        "color": (0, 200, 200),          # Yellow (BGR)
        "bg_color": (0, 80, 80),
        "instruction": "Recyclable — Handle with care!",
        "details": "Bottles, jars (rinse first)",
        "icon": "🫙",
    },
    "METAL": {
        "bin": "🟡 YELLOW BIN",
        "color": (0, 165, 255),          # Orange
        "bg_color": (0, 60, 100),
        "instruction": "Recyclable — Crush if possible",
        "details": "Cans, aluminum, tin, foil",
        "icon": "🥫",
    },
    "PAPER": {
        "bin": "🔵 BLUE BIN",
        "color": (255, 150, 50),         # Light blue
        "bg_color": (100, 50, 0),
        "instruction": "Recyclable — Keep dry!",
        "details": "Newspapers, magazines, office paper",
        "icon": "📄",
    },
    "PLASTIC": {
        "bin": "🟡 YELLOW BIN",
        "color": (0, 100, 255),          # Red
        "bg_color": (0, 30, 100),
        "instruction": "Recyclable — Check resin code",
        "details": "Bottles (PET), containers (HDPE)",
        "icon": "♻️",
    },
}

# Detection statistics
stats = {
    "total_detections": 0,
    "class_counts": {},
    "start_time": None,
    "fps_history": [],
}


def find_best_weights():
    """Find the best trained model weights."""
    project_root = Path(__file__).parent.parent
    possible_paths = [
        project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
        project_root / "models" / "best.pt",
        project_root / "best.pt",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    # Search recursively for any best.pt
    for pt_file in project_root.rglob("best.pt"):
        return str(pt_file)

    return None


def draw_sorting_panel(frame, detections, conf_threshold):
    """Draw the sorting recommendation panel on the right side of the frame."""
    h, w = frame.shape[:2]
    panel_width = 340
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)

    # Panel background — dark gradient
    for y in range(h):
        intensity = int(25 + (y / h) * 10)
        panel[y, :] = (intensity, intensity, intensity + 5)

    # Header
    cv2.rectangle(panel, (0, 0), (panel_width, 55), (40, 40, 45), -1)
    cv2.putText(panel, "WASTE SORTING SYSTEM", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
    cv2.putText(panel, f"Confidence: {conf_threshold:.0%}", (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Separator
    cv2.line(panel, (10, 58), (panel_width - 10, 58), (60, 60, 60), 1)

    if detections:
        y_offset = 75
        for i, det in enumerate(detections[:4]):  # Show top 4 detections
            cls_name = det["class"]
            confidence = det["confidence"]
            sort_info = SORTING_MAP.get(cls_name, {
                "bin": "❓ UNKNOWN",
                "color": (150, 150, 150),
                "bg_color": (50, 50, 50),
                "instruction": "Manual sorting required",
                "details": "",
                "icon": "❓",
            })

            # Detection card background
            card_h = 100
            cv2.rectangle(panel, (8, y_offset), (panel_width - 8, y_offset + card_h),
                          sort_info["bg_color"], -1)
            cv2.rectangle(panel, (8, y_offset), (panel_width - 8, y_offset + card_h),
                          sort_info["color"], 1)

            # Left accent bar
            cv2.rectangle(panel, (8, y_offset), (14, y_offset + card_h),
                          sort_info["color"], -1)

            # Class name + confidence
            cv2.putText(panel, f"{cls_name}", (22, y_offset + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Confidence bar
            bar_x = 22
            bar_y = y_offset + 32
            bar_w = panel_width - 45
            bar_h = 12
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          (50, 50, 50), -1)
            fill_w = int(bar_w * confidence)
            # Color the bar based on confidence
            if confidence > 0.7:
                bar_color = (0, 200, 0)
            elif confidence > 0.4:
                bar_color = (0, 200, 200)
            else:
                bar_color = (0, 0, 200)
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          bar_color, -1)
            cv2.putText(panel, f"{confidence:.1%}", (bar_x + bar_w + 2, bar_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

            # Bin recommendation
            cv2.putText(panel, sort_info["bin"], (22, y_offset + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, sort_info["color"], 2)

            # Instruction
            cv2.putText(panel, sort_info["instruction"], (22, y_offset + 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

            # Details
            cv2.putText(panel, sort_info["details"], (22, y_offset + 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (130, 130, 130), 1)

            y_offset += card_h + 10
    else:
        # No detections message
        cv2.putText(panel, "No waste detected", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)
        cv2.putText(panel, "Place items in front", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        cv2.putText(panel, "of the camera", (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    # Statistics section at bottom
    stats_y = h - 140
    cv2.line(panel, (10, stats_y), (panel_width - 10, stats_y), (60, 60, 60), 1)
    cv2.putText(panel, "SESSION STATS", (15, stats_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

    cv2.putText(panel, f"Total Detections: {stats['total_detections']}",
                (15, stats_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Show top classes detected
    if stats["class_counts"]:
        sorted_classes = sorted(stats["class_counts"].items(), key=lambda x: x[1], reverse=True)
        y_stat = stats_y + 65
        for cls, count in sorted_classes[:4]:
            color = SORTING_MAP.get(cls, {}).get("color", (150, 150, 150))
            cv2.putText(panel, f"{cls}: {count}", (15, y_stat),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_stat += 18

    # Controls hint
    cv2.putText(panel, "[Q]uit [S]creenshot [P]ause [+/-] Conf",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

    return panel


def draw_detections(frame, results, conf_threshold):
    """Draw bounding boxes and labels on the frame."""
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id].upper()

            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            sort_info = SORTING_MAP.get(cls_name, {
                "color": (150, 150, 150),
                "bg_color": (50, 50, 50),
            })
            color = sort_info["color"]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw filled label background
            label = f"{cls_name} {conf:.0%}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 8, y1),
                          color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Corner accents
            corner_len = 20
            thickness = 3
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "box": (x1, y1, x2, y2),
            })

            # Update stats
            stats["total_detections"] += 1
            stats["class_counts"][cls_name] = stats["class_counts"].get(cls_name, 0) + 1

    return frame, detections


def draw_fps(frame, fps):
    """Draw FPS counter on frame."""
    h, w = frame.shape[:2]
    fps_text = f"FPS: {fps:.1f}"

    # Background
    cv2.rectangle(frame, (10, 10), (130, 40), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (130, 40), (100, 200, 255), 1)

    # FPS with color based on performance
    if fps > 25:
        fps_color = (0, 255, 0)
    elif fps > 15:
        fps_color = (0, 200, 200)
    else:
        fps_color = (0, 0, 255)

    cv2.putText(frame, fps_text, (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

    return frame


def run_detection(args):
    """Main detection loop."""
    print()
    print("🗑️  PLASTIC WASTE SORTING SYSTEM — REAL-TIME DETECTION")
    print("=" * 60)

    # Load model
    weights = args.weights
    if weights is None:
        weights = find_best_weights()
        if weights is None:
            print("  ❌ No trained model found!")
            print("  💡 Train first: python src/train.py")
            sys.exit(1)

    print(f"  ✅ Loading model: {weights}")
    model = YOLO(weights)
    print(f"  ✅ Model loaded successfully")

    # Open webcam
    print(f"  📹 Opening camera source: {args.source}")
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)

    if not cap.isOpened():
        print("  ❌ Cannot open camera!")
        print("  💡 Try: python src/detect.py --source 1")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  ✅ Camera opened: {actual_w}x{actual_h}")
    print()
    print("  🎮 Controls:")
    print("     Q/ESC — Quit")
    print("     S     — Screenshot")
    print("     P     — Pause/Resume")
    print("     +/-   — Adjust confidence threshold")
    print("     R     — Reset statistics")
    print()
    print("  Starting detection... (press Q to quit)")
    print()

    conf_threshold = args.conf
    paused = False
    stats["start_time"] = time.time()
    frame_count = 0
    fps = 0.0
    fps_timer = time.time()

    # Create output directory for screenshots
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("  ⚠️  Failed to grab frame")
                break

            # Run inference
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                iou=0.45,
                verbose=False,
                stream=False,
            )

            # Draw detections
            frame, detections = draw_detections(frame, results, conf_threshold)

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            # Draw FPS
            frame = draw_fps(frame, fps)

            # Draw sorting panel
            panel = draw_sorting_panel(frame, detections, conf_threshold)

            # Combine frame and panel
            display = np.hstack([frame, panel])

        else:
            # Paused — show pause indicator
            cv2.putText(frame, "PAUSED", (actual_w // 2 - 80, actual_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            panel = draw_sorting_panel(frame, detections, conf_threshold)
            display = np.hstack([frame, panel])

        # Show window
        cv2.imshow("Waste Sorting System", display)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # Q or ESC
            break
        elif key == ord("s"):  # Screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = output_dir / f"detection_{timestamp}.jpg"
            cv2.imwrite(str(screenshot_path), display)
            print(f"  📸 Screenshot saved: {screenshot_path}")
        elif key == ord("p"):  # Pause
            paused = not paused
            print(f"  {'⏸️  Paused' if paused else '▶️  Resumed'}")
        elif key == ord("+") or key == ord("="):  # Increase threshold
            conf_threshold = min(0.95, conf_threshold + 0.05)
            print(f"  🎯 Confidence threshold: {conf_threshold:.0%}")
        elif key == ord("-"):  # Decrease threshold
            conf_threshold = max(0.05, conf_threshold - 0.05)
            print(f"  🎯 Confidence threshold: {conf_threshold:.0%}")
        elif key == ord("r"):  # Reset stats
            stats["total_detections"] = 0
            stats["class_counts"] = {}
            stats["start_time"] = time.time()
            print("  🔄 Statistics reset")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Print session summary
    print()
    print("=" * 60)
    print("📊 SESSION SUMMARY")
    print("=" * 60)
    duration = time.time() - stats["start_time"]
    print(f"  ⏱️  Duration: {duration:.0f}s")
    print(f"  🔢 Total Detections: {stats['total_detections']}")
    if stats["class_counts"]:
        print("  📋 Breakdown:")
        for cls, count in sorted(stats["class_counts"].items(), key=lambda x: x[1], reverse=True):
            print(f"      {cls}: {count}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Waste Detection & Sorting Recommendation"
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to trained model weights (auto-detected if not specified)"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Camera source — 0 for default webcam, 1 for external (default: 0)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Confidence threshold (default: 0.35)"
    )

    args = parser.parse_args()
    run_detection(args)


if __name__ == "__main__":
    main()
