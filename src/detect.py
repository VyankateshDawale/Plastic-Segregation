"""
detect.py — Grand Unified Adaptive Architecture (Real-Time Inference)
=====================================================================

This script fuses YOLO visual detection with the newly implemented:
1. Dynamic Confidence Gating (Idea 1)
2. Expected Failure Probability (EFP) Prediction (Idea 2)
3. Simulated Fusion Transformer / Modality Routing (Idea 3)
4. Out-of-Distribution (OOD) Anomaly Detection

Usage:
    python src/detect.py
"""

import argparse
import cv2
import numpy as np
import time
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import joblib

# ══════════════════════════════════════════════════════════════
# ADVANCED ARCHITECTURE GLOBALS
# ══════════════════════════════════════════════════════════════

# Try loading the EFP Predictor we trained earlier
PROJECT_ROOT = Path(__file__).parent.parent
EFP_MODEL_PATH = PROJECT_ROOT / "models" / "efp_predictor.pkl"
EFP_MODEL = None
if EFP_MODEL_PATH.exists():
    try:
        EFP_MODEL = joblib.load(EFP_MODEL_PATH)
        print(f"  [+] Loaded Expected Failure Probability AI from {EFP_MODEL_PATH}")
    except Exception as e:
        print(f"  [-] Failed to load EFP AI: {e}")

# Sorting Map with specific hazard/anomaly bin
SORTING_MAP = {
    "BIODEGRADABLE": {"bin": "🟢 GREEN BIN", "color": (0, 200, 0), "bg": (0, 80, 0)},
    "CARDBOARD": {"bin": "🔵 BLUE BIN", "color": (200, 150, 0), "bg": (80, 50, 0)},
    "GLASS": {"bin": "🟡 YELLOW BIN", "color": (0, 200, 200), "bg": (0, 80, 80)},
    "METAL": {"bin": "🟡 YELLOW BIN", "color": (0, 165, 255), "bg": (0, 60, 100)},
    "PAPER": {"bin": "🔵 BLUE BIN", "color": (255, 150, 50), "bg": (100, 50, 0)},
    "PLASTIC": {"bin": "🟡 YELLOW BIN", "color": (0, 100, 255), "bg": (0, 30, 100)},
    "ANOMALY": {"bin": "🔴 HAZARD/REJECT", "color": (0, 0, 255), "bg": (50, 0, 0)}
}

# ══════════════════════════════════════════════════════════════
# ANALYTICS & EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_visual_features(crop):
    """Extract real-time visual features for the EFP Predictor."""
    if crop is None or crop.size == 0:
        return 0.5, 0.5, 0.5
        
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 1. Darkness (0 to 1) - proxy for carbon black
    mean_val = np.mean(gray)
    darkness = np.clip(1.0 - (mean_val / 255.0), 0.0, 1.0)
    
    # 2. Gloss Index (Standard deviation of brightness indicates specular highlights)
    std_val = np.std(gray)
    gloss = np.clip(std_val / 128.0, 0.0, 1.0)
    
    # 3. Texture Roughness (Laplacian Variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture = np.clip(laplacian_var / 1000.0, 0.0, 1.0)
    
    return darkness, gloss, texture

def compute_dynamic_threshold(base_conf, ambient_lux=400, camera_temp=45.0):
    """Idea 1: Dynamic Self-Learning Confidence Gate."""
    adj = 0.0
    if ambient_lux < 500: adj += 0.05
    if camera_temp > 40: adj += 0.03
    return min(0.95, base_conf + adj)

def calculate_anomaly_score(darkness, texture):
    """OOD Anomaly Detection: Flags items completely outside plastic distributions."""
    # Simulated Mahalanobis distance proxy
    score = (darkness * 0.8) + (texture * 0.4) + np.random.uniform(0, 0.1)
    return score

# ══════════════════════════════════════════════════════════════
# GUI AND DRAWING
# ══════════════════════════════════════════════════════════════

def draw_sorting_panel(frame, detections, dynamic_thresh):
    h, w = frame.shape[:2]
    panel_width = 450
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)

    # Background Gradient
    for y in range(h):
        intensity = int(20 + (y / h) * 15)
        panel[y, :] = (intensity, intensity, intensity + 5)

    # Header
    cv2.rectangle(panel, (0, 0), (panel_width, 60), (30, 30, 35), -1)
    cv2.putText(panel, "UNIFIED MULTIMODAL PIPELINE", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 255), 2)
    cv2.putText(panel, f"Dynamic Gate Target: {dynamic_thresh:.1%}", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.line(panel, (10, 60), (panel_width - 10, 60), (80, 80, 80), 1)

    if detections:
        y_offset = 75
        for det in detections[:3]:  # Top 3
            cls_name = det["class"]
            conf = det["confidence"]
            efp = det["efp"]
            active_sensor = det["sensor"]
            is_anomaly = det["anomaly"]
            
            # Map styling
            style = SORTING_MAP.get(cls_name if not is_anomaly else "ANOMALY", SORTING_MAP["ANOMALY"])
            
            card_h = 130
            cv2.rectangle(panel, (10, y_offset), (panel_width - 10, y_offset + card_h), style["bg"], -1)
            cv2.rectangle(panel, (10, y_offset), (panel_width - 10, y_offset + card_h), style["color"], 1)
            cv2.rectangle(panel, (10, y_offset), (16, y_offset + card_h), style["color"], -1) # Left accent

            # Main Class & Conf
            cv2.putText(panel, f"{'⚠️ ANOMALY' if is_anomaly else cls_name}", (25, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(panel, f"YOLO Softmax: {conf:.1%}", (220, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # EFP Score & Routing
            efp_color = (0, 0, 255) if efp > 0.6 else (0, 255, 0)
            cv2.putText(panel, f"EFP Predictor: {efp:.1%}", (25, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, efp_color, 1)
            
            # Sensor Fusion Flow
            cv2.putText(panel, "Fusion Routing:", (25, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            cv2.putText(panel, active_sensor, (140, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Destination Bin
            cv2.putText(panel, f"Destination: {style['bin']}", (25, y_offset + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, style["color"], 2)

            y_offset += card_h + 10
    else:
        cv2.putText(panel, "Awaiting Input...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    return panel

def draw_detections_and_fuse(frame, results, base_conf, dynamic_thresh):
    detections = []
    
    for result in results:
        if result.boxes is None: continue
        for box in result.boxes:
            conf = float(box.conf[0])
            
            # Use Dynamic Threshold instead of Static! (Idea 1)
            if conf < dynamic_thresh * 0.5: # We still process low conf items to check for anomalies/MIR routing
                continue
                
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id].upper()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop image and extract multimodal proxy features
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            darkness, gloss, texture = extract_visual_features(crop)
            obj_size = ((x2 - x1) * (y2 - y1)) / 1000.0
            
            # OOD Anomaly Check
            anomaly_score = calculate_anomaly_score(darkness, texture)
            is_anomaly = anomaly_score > 0.85
            
            # Expected Failure Probability Prediction (Idea 2)
            efp = 0.0
            if EFP_MODEL and not is_anomaly:
                # Features: rgb_darkness, nir_baseline_intensity, lighting_lux, gloss_index, texture_roughness, object_size_cm
                sim_nir_base = max(0.01, 1.0 - (darkness * np.random.uniform(0.8, 1.2)))
                X = pd.DataFrame([{
                    'rgb_darkness': darkness,
                    'nir_baseline_intensity': sim_nir_base,
                    'lighting_lux': 500,
                    'gloss_index': gloss,
                    'texture_roughness': texture,
                    'object_size_cm': obj_size
                }])
                efp = float(EFP_MODEL.predict_proba(X)[0][1])
            else:
                # Simulated EFP if model missing
                efp = min(1.0, darkness * 0.6 + texture * 0.4)
                
            # Idea 3: Sensor Fusion Routing Engine
            active_sensor = "RGB CAM"
            if is_anomaly:
                active_sensor = "REJECTED (OOD HAZARD)"
            elif cls_name == "PLASTIC":
                if efp > 0.60:
                    active_sensor = "RGB -> MIR (Bypassed NIR)"
                elif conf < dynamic_thresh:
                    active_sensor = "RGB -> NIR -> MIR"
                else:
                    active_sensor = "RGB -> NIR"
                    
            color = SORTING_MAP["ANOMALY"]["color"] if is_anomaly else SORTING_MAP.get(cls_name, {"color": (150, 150, 150)})["color"]
            
            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{'ANOMALY' if is_anomaly else cls_name} ({active_sensor})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "efp": efp,
                "sensor": active_sensor,
                "anomaly": is_anomaly
            })
            
    return frame, detections

def find_best_weights():
    for p in [PROJECT_ROOT/"runs"/"detect"/"train"/"weights"/"best.pt", PROJECT_ROOT/"models"/"best.pt", PROJECT_ROOT/"best.pt"]:
        if p.exists(): return str(p)
    return None

def run_detection(args):
    print("\n🗑️  ADVANCED MULTIMODAL WASTE SEGREGATION (UNIFIED ARCHITECTURE)")
    print("=" * 70)
    
    weights = args.weights or find_best_weights()
    if not weights:
        print("  ❌ No trained YOLO model found! Please train the model first.")
        sys.exit(1)

    model = YOLO(weights)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    base_conf = args.conf
    print(f"  ✅ Camera loaded. Running Deep Fusion Pipeline... (Q to quit)\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Calculate dynamic threshold based on simulated environment
        # e.g., simulating low light by dropping ambient_lux randomly
        dynamic_thresh = compute_dynamic_threshold(base_conf, ambient_lux=np.random.uniform(400, 800))

        results = model.predict(frame, conf=0.1, iou=0.45, verbose=False)
        frame, detections = draw_detections_and_fuse(frame, results, base_conf, dynamic_thresh)
        
        panel = draw_sorting_panel(frame, detections, dynamic_thresh)
        display = np.hstack([frame, panel])

        cv2.imshow("Unified Architecture Pipeline", display)
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.70)
    run_detection(parser.parse_args())
