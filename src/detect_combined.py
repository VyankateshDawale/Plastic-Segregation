"""
Combined Waste Detection + Classification Pipeline
===================================================
1. Detection model (YOLOv11s) → finds waste objects with bounding boxes
2. Classification model (YOLOv11s-cls) → classifies each detected crop (95.7% accuracy)
3. Displays sorting recommendations for each detected object
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO

# Model paths
DETECT_MODEL = r"D:\projects\Plastic Waste Sorting System\runs\detect\runs\detect\train\weights\best.pt"
CLASSIFY_MODEL = r"D:\projects\Plastic Waste Sorting System\runs\classify\train\weights\best.pt"

# Classification classes (10 classes)
CLS_CLASSES = ["battery", "biological", "cardboard", "clothes", "glass", 
               "metal", "paper", "plastic", "shoes", "trash"]

# Sorting recommendations for each class
SORTING_INFO = {
    "battery":       {"bin": "HAZARDOUS WASTE",  "color": (0, 0, 200),    "icon": "!!",  "tip": "Take to e-waste collection point"},
    "biological":    {"bin": "GREEN BIN (WET)",   "color": (0, 180, 0),    "icon": "**",  "tip": "Compostable - organic waste bin"},
    "cardboard":     {"bin": "BLUE BIN (DRY)",    "color": (200, 150, 0),  "icon": "##",  "tip": "Flatten & recycle - dry waste"},
    "clothes":       {"bin": "DONATION / TEXTILE","color": (180, 0, 180),  "icon": "~~",  "tip": "Donate if reusable, else textile bin"},
    "glass":         {"bin": "BLUE BIN (DRY)",    "color": (0, 200, 200),  "icon": "[]",  "tip": "Rinse & recycle - handle carefully"},
    "metal":         {"bin": "BLUE BIN (DRY)",    "color": (100, 100, 255),"icon": "<>",  "tip": "Rinse & recycle - crush if possible"},
    "paper":         {"bin": "BLUE BIN (DRY)",    "color": (255, 200, 100),"icon": "==",  "tip": "Keep dry & recycle"},
    "plastic":       {"bin": "BLUE BIN (DRY)",    "color": (0, 165, 255),  "icon": "@@",  "tip": "Rinse & recycle by type"},
    "shoes":         {"bin": "DONATION / TEXTILE","color": (150, 100, 50), "icon": "^^",  "tip": "Donate if wearable, else textile bin"},
    "trash":         {"bin": "RED BIN (REJECT)",  "color": (80, 80, 80),   "icon": "XX",  "tip": "Non-recyclable - goes to landfill"},

    # Detection model classes (fallback)
    "BIODEGRADABLE": {"bin": "GREEN BIN (WET)",   "color": (0, 180, 0),    "icon": "**",  "tip": "Compostable - organic waste bin"},
    "CARDBOARD":     {"bin": "BLUE BIN (DRY)",    "color": (200, 150, 0),  "icon": "##",  "tip": "Flatten & recycle - dry waste"},
    "GLASS":         {"bin": "BLUE BIN (DRY)",    "color": (0, 200, 200),  "icon": "[]",  "tip": "Rinse & recycle - handle carefully"},
    "METAL":         {"bin": "BLUE BIN (DRY)",    "color": (100, 100, 255),"icon": "<>",  "tip": "Rinse & recycle - crush if possible"},
    "PAPER":         {"bin": "BLUE BIN (DRY)",    "color": (255, 200, 100),"icon": "==",  "tip": "Keep dry & recycle"},
    "PLASTIC":       {"bin": "BLUE BIN (DRY)",    "color": (0, 165, 255),  "icon": "@@",  "tip": "Rinse & recycle by type"},
}


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=10):
    """Draw a rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_label(img, text, pos, color, bg_alpha=0.7):
    """Draw text with semi-transparent background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    
    # Background
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 2, y - h - 8), (x + w + 6, y + 4), color, -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    
    # Text
    cv2.putText(img, text, (x + 2, y - 4), font, scale, (255, 255, 255), thickness)


def create_info_panel(detections, panel_width=380, panel_height=720):
    """Create a side panel showing sorting recommendations."""
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 35)  # Dark background
    
    # Header
    cv2.rectangle(panel, (0, 0), (panel_width, 55), (50, 50, 60), -1)
    cv2.putText(panel, "WASTE SORTING SYSTEM", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    cv2.putText(panel, "Detection + Classification Pipeline", (15, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Divider
    cv2.line(panel, (10, 60), (panel_width - 10, 60), (80, 80, 80), 1)
    
    if not detections:
        cv2.putText(panel, "No waste detected", (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)
        cv2.putText(panel, "Point camera at waste items", (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    else:
        y_offset = 75
        for i, det in enumerate(detections[:5]):  # Show max 5 detections
            cls_name = det["class"]
            det_conf = det["det_conf"]
            cls_conf = det["cls_conf"]
            info = SORTING_INFO.get(cls_name, SORTING_INFO.get("trash"))
            color = info["color"]
            
            # Detection box
            cv2.rectangle(panel, (10, y_offset), (panel_width - 10, y_offset + 120), (45, 45, 50), -1)
            cv2.rectangle(panel, (10, y_offset), (panel_width - 10, y_offset + 120), color, 2)
            
            # Class name with icon
            cv2.putText(panel, f"{info['icon']} {cls_name.upper()}", (20, y_offset + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Confidence bars
            cv2.putText(panel, f"Detect:", (20, y_offset + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            bar_w = int(200 * det_conf)
            cv2.rectangle(panel, (90, y_offset + 35), (90 + bar_w, y_offset + 48), color, -1)
            cv2.putText(panel, f"{det_conf:.0%}", (300, y_offset + 47),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            
            cv2.putText(panel, f"Class:", (20, y_offset + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            bar_w = int(200 * cls_conf)
            cv2.rectangle(panel, (90, y_offset + 55), (90 + bar_w, y_offset + 68), (0, 220, 100), -1)
            cv2.putText(panel, f"{cls_conf:.0%}", (300, y_offset + 67),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            
            # Bin recommendation
            cv2.putText(panel, f"BIN: {info['bin']}", (20, y_offset + 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)
            
            # Tip
            cv2.putText(panel, info["tip"], (20, y_offset + 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (140, 140, 140), 1)
            
            y_offset += 135
    
    # Footer
    cv2.line(panel, (10, panel_height - 40), (panel_width - 10, panel_height - 40), (80, 80, 80), 1)
    cv2.putText(panel, "Press 'Q' to quit | 'S' to screenshot", (15, panel_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    
    return panel


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Combined Waste Detection + Classification")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0, 1, 2) or IP camera URL")
    args = parser.parse_args()

    print("=" * 60)
    print("  COMBINED WASTE DETECTION + CLASSIFICATION")
    print("=" * 60)
    
    # Load models
    print("\n  Loading Detection Model...")
    det_model = YOLO(DETECT_MODEL)
    print(f"  >> Loaded: {DETECT_MODEL}")
    
    print("  Loading Classification Model...")
    cls_model = YOLO(CLASSIFY_MODEL)
    print(f"  >> Loaded: {CLASSIFY_MODEL}")
    
    # Parse source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Open webcam
    print(f"\n  Opening camera source: {source}...")
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("  ERROR: Cannot open webcam!")
        return
    
    print("  Webcam ready! Press 'Q' to quit.\n")
    
    fps_counter = 0
    fps_time = time.time()
    fps_display = 0
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        detections = []
        
        # Step 1: Run detection model
        det_results = det_model(frame, conf=0.3, verbose=False)
        
        if det_results and len(det_results[0].boxes) > 0:
            boxes = det_results[0].boxes
            
            for box in boxes:
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                det_conf = float(box.conf[0])
                det_class = det_results[0].names[int(box.cls[0])]
                
                # Clamp coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Step 2: Crop detected region and classify
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                cls_results = cls_model(crop, verbose=False)
                cls_conf = float(cls_results[0].probs.top1conf)
                cls_class = cls_results[0].names[int(cls_results[0].probs.top1)]
                
                # Use classification result (more accurate with 10 classes)
                final_class = cls_class
                info = SORTING_INFO.get(final_class, SORTING_INFO.get("trash"))
                color = info["color"]
                
                detections.append({
                    "class": final_class,
                    "det_conf": det_conf,
                    "cls_conf": cls_conf,
                    "det_class": det_class,
                    "box": (x1, y1, x2, y2)
                })
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw corner accents
                corner_len = 20
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 4)
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4)
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)
                
                # Label with class and confidence
                label = f"{final_class.upper()} {cls_conf:.0%}"
                draw_label(frame, label, (x1, y1 - 2), color)
                
                # Bin recommendation below box
                bin_label = f"-> {info['bin']}"
                draw_label(frame, bin_label, (x1, y2 + 22), (50, 50, 50), bg_alpha=0.6)
        
        # FPS counter
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        # FPS overlay
        cv2.putText(frame, f"FPS: {fps_display}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Objects count
        cv2.putText(frame, f"Objects: {len(detections)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Create info panel
        panel = create_info_panel(detections, panel_height=frame.shape[0])
        
        # Combine frame and panel
        combined = np.hstack([frame, panel])
        
        cv2.imshow("Waste Sorting System - Combined Pipeline", combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            screenshot_count += 1
            filename = f"D:\\projects\\Plastic Waste Sorting System\\outputs\\screenshot_{screenshot_count}.jpg"
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            cv2.imwrite(filename, combined)
            print(f"  Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n  Webcam closed. Goodbye!")


if __name__ == "__main__":
    main()
