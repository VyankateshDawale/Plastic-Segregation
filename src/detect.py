"""
detect.py — Grand Unified Adaptive Architecture (Real-Time Inference)
=====================================================================

Wires the following REAL trained models and deterministic logic into the live webcam feed:
  1. ACE Engine (XGBoost, models/ace_xgboost.json) — Dynamic Confidence Gating (Idea 1)
  2. EFP Predictor (RandomForest, models/efp_predictor.pkl) — Expected Failure Probability (Idea 2)
  3. Cross-Modal Attention routing logic — Sensor Fusion Engine (Idea 3)
  4. Deterministic OOD Anomaly Detection — no random noise, physics-based scoring

Usage:
    python src/detect.py [--weights PATH] [--source 0] [--conf 0.70]
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import joblib

# ── Make project root importable so `src.ace` works ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

# ── Try loading the ACE Engine (Idea 1: Dynamic Confidence Gate) ─────────────
ACE_ENGINE = None
ACE_MODEL_PATH = PROJECT_ROOT / "models" / "ace_xgboost.json"
try:
    from src.ace.engine import ACEEngine
    if ACE_MODEL_PATH.exists():
        ACE_ENGINE = ACEEngine(model_path=str(ACE_MODEL_PATH))
        print(f"  [✓] ACE Engine loaded → Dynamic confidence gate ACTIVE")
    else:
        print(f"  [!] ace_xgboost.json not found — falling back to fixed threshold")
except Exception as e:
    print(f"  [!] ACE Engine import failed: {e}")

# ── Try loading the EFP Predictor (Idea 2: Expected Failure Probability) ──────
EFP_MODEL = None
EFP_MODEL_PATH = PROJECT_ROOT / "models" / "efp_predictor.pkl"
try:
    if EFP_MODEL_PATH.exists():
        EFP_MODEL = joblib.load(EFP_MODEL_PATH)
        print(f"  [✓] EFP Predictor loaded → NIR bypass prediction ACTIVE")
    else:
        print(f"  [!] efp_predictor.pkl not found — using fallback EFP formula")
except Exception as e:
    print(f"  [!] EFP model load failed: {e}")

# ── Bin destination map ───────────────────────────────────────────────────────
SORTING_MAP = {
    "BIODEGRADABLE": {"bin": "GREEN BIN",   "color": (50, 200, 50),   "bg": (0, 60, 0)},
    "CARDBOARD":     {"bin": "BLUE BIN",    "color": (210, 160, 30),  "bg": (70, 45, 0)},
    "GLASS":         {"bin": "YELLOW BIN",  "color": (50, 210, 210),  "bg": (0, 70, 70)},
    "METAL":         {"bin": "YELLOW BIN",  "color": (50, 170, 255),  "bg": (0, 50, 90)},
    "PAPER":         {"bin": "BLUE BIN",    "color": (255, 160, 60),  "bg": (90, 45, 0)},
    "PLASTIC":       {"bin": "YELLOW BIN",  "color": (60, 120, 255),  "bg": (0, 25, 90)},
    "TRASH":         {"bin": "BLACK BIN",   "color": (160, 160, 160), "bg": (30, 30, 30)},
    "ANOMALY":       {"bin": "REJECT/HAZARD","color": (50, 50, 255),  "bg": (60, 0, 0)},
}

# ── Polymer class index used by ACE feature_extractor ─────────────────────────
POLYMER_IDX = {"PET": 0, "HDPE": 1, "PVC": 2, "LDPE": 3, "PP": 4, "PS": 5}

# Recent MIR escalation tracker (rolling 60-item window)
_recent_escalations: list[int] = []


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (deterministic — no random noise)
# ══════════════════════════════════════════════════════════════════════════════

def extract_visual_features(crop: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Returns (darkness, gloss, texture_roughness, hue_mean, value_mean)
    All values clipped to [0, 1].

    Physics rationale:
      - darkness  : 1 - mean brightness.  High → carbon-black candidate.
      - gloss     : std-dev of brightness / 128. Plastic has specular highlights → high gloss.
      - texture   : Laplacian variance / 1000. Metal/wood have high edge energy.
      - hue_mean  : HSV hue channel mean (0-180 → 0-1).
      - value_mean: HSV value channel mean (brightness, 0-1).
    """
    if crop is None or crop.size == 0:
        return 0.5, 0.5, 0.5, 0.5, 0.5

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    mean_v   = float(np.mean(gray))
    darkness = float(np.clip(1.0 - mean_v / 255.0,  0.0, 1.0))
    gloss    = float(np.clip(np.std(gray) / 128.0,  0.0, 1.0))
    texture  = float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0, 0.0, 1.0))
    hue_mean = float(np.mean(hsv[:, :, 0]) / 180.0)
    val_mean = float(np.mean(hsv[:, :, 2]) / 255.0)

    return darkness, gloss, texture, hue_mean, val_mean


def compute_ambient_lux(frame: np.ndarray) -> float:
    """
    Estimate scene brightness in lux-proxy from the full frame's V-channel mean.
    Maps [0, 255] → [0, 1200] lux (rough linear approximation).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_brightness = float(np.mean(hsv[:, :, 2]))
    return float(np.clip((mean_brightness / 255.0) * 1200.0, 50.0, 1200.0))


# ══════════════════════════════════════════════════════════════════════════════
# IDEA 1 — Dynamic Confidence Gate via ACE Engine
# ══════════════════════════════════════════════════════════════════════════════

def get_dynamic_threshold(
    base_conf: float,
    darkness: float,
    gloss: float,
    texture: float,
    hue: float,
    val: float,
    obj_size_px: float,
    ambient_lux: float,
    cls_name: str,
) -> float:
    """
    Uses the real ACE XGBoost model to predict whether MIR escalation is needed,
    then maps that probability to a dynamic confidence threshold.

    Falls back to a deterministic formula if ACE model is unavailable.

    ACE feature vector (15 features, same order as feature_extractor.py):
      nir_confidence, nir_margin, reflectance_mean, reflectance_var,
      spectral_entropy, obj_hue_mean, obj_value_mean, obj_size_px,
      obj_aspect_ratio, ambient_light_lux, sensor_temp_c, humidity_pct,
      conveyor_speed_ms, predicted_class_encoded, recent_mir_rate
    """
    recent_mir_rate = (sum(_recent_escalations[-60:]) / max(len(_recent_escalations[-60:]), 1))

    # Derive proxy NIR features from visual features (no real NIR sensor in webcam mode)
    # reflectance_mean proxy: dark objects have low reflectance
    reflectance_mean   = float(np.clip(1.0 - darkness, 0.01, 1.0))
    reflectance_var    = reflectance_mean * 0.05        # smooth surfaces → low variance
    nir_conf_proxy     = float(np.clip(reflectance_mean * 1.1, 0.05, 0.99))
    nir_margin_proxy   = nir_conf_proxy * 0.20
    spectral_entropy   = float(np.clip(2.5 + darkness * 3.0, 1.0, 6.0))  # dark → low entropy

    cls_enc = float(POLYMER_IDX.get(cls_name.upper(), 0))

    features = np.array([
        nir_conf_proxy,         # nir_confidence
        nir_margin_proxy,       # nir_confidence_margin
        reflectance_mean,       # reflectance_mean
        reflectance_var,        # reflectance_variance
        spectral_entropy,       # spectral_entropy
        hue * 180.0,            # object_hue_mean
        val * 255.0,            # object_value_mean
        obj_size_px,            # object_size_px
        1.0,                    # object_aspect_ratio (unknown → neutral)
        ambient_lux,            # ambient_light_lux
        28.0,                   # sensor_temperature_c (room temp assumed)
        55.0,                   # humidity_pct (nominal)
        0.25,                   # conveyor_speed_ms (nominal)
        cls_enc,                # predicted_class_encoded
        recent_mir_rate,        # recent_mir_rate
    ], dtype=np.float32)

    if ACE_ENGINE is not None:
        try:
            return float(ACE_ENGINE.predict_threshold(features))
        except Exception:
            pass  # fall through to deterministic fallback

    # Deterministic fallback (Idea 1 formula without random noise)
    adj = 0.0
    if ambient_lux < 300:  adj += 0.06
    elif ambient_lux < 500: adj += 0.03
    if darkness > 0.7:      adj += 0.04   # very dark object → stricter gate
    if texture > 0.6:       adj += 0.02   # high texture → uncertain material
    return float(min(0.95, base_conf + adj))


# ══════════════════════════════════════════════════════════════════════════════
# IDEA 2 — Expected Failure Probability (EFP) Predictor
# ══════════════════════════════════════════════════════════════════════════════

def compute_efp(darkness: float, gloss: float, texture: float, obj_size_px: float, ambient_lux: float) -> float:
    """
    Runs the trained EFP RandomForest model. If unavailable, uses the
    deterministic physics formula:
        P_fail ≈ darkness (NIR absorptance proxy) − gloss (surface reflection bonus)

    EFP model feature order (from scripts/train_efp_model.py):
      rgb_darkness, nir_baseline_intensity, lighting_lux,
      gloss_index, texture_roughness, object_size_cm
    """
    # Simulate NIR baseline: dark objects have near-zero baseline
    nir_baseline = float(np.clip(1.0 - darkness * 1.05, 0.01, 1.0))
    obj_size_cm  = obj_size_px / 100.0  # rough pixel-to-cm proxy

    if EFP_MODEL is not None:
        try:
            import pandas as pd
            X = pd.DataFrame([{
                "rgb_darkness":         darkness,
                "nir_baseline_intensity": nir_baseline,
                "lighting_lux":         ambient_lux,
                "gloss_index":          gloss,
                "texture_roughness":    texture,
                "object_size_cm":       obj_size_cm,
            }])
            return float(EFP_MODEL.predict_proba(X)[0][1])
        except Exception:
            pass  # fall through to formula

    # Deterministic fallback formula (no random noise)
    p_fail = (darkness * 0.65) + (texture * 0.20) - (gloss * 0.15)
    return float(np.clip(p_fail, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# OOD ANOMALY DETECTION  (deterministic — no np.random)
# ══════════════════════════════════════════════════════════════════════════════

def compute_anomaly_score(darkness: float, texture: float, gloss: float) -> float:
    """
    Physics-based Out-of-Distribution score.

    Plastic signatures:
      - Moderate to high gloss (smooth surface)
      - Low-to-medium texture (uniform surface)
      - Variable darkness (depends on colour/pigment)

    Non-plastic anomalies (metal, wood, glass shards, batteries):
      - Metal:  very high texture + high gloss, moderate darkness
      - Wood:   very high texture + low gloss, moderate darkness
      - Battery: very high texture + low gloss + very dark

    Score formula:  high texture AND low gloss → strongly non-plastic
    Threshold chosen at 0.70 (generous) to avoid false positives on
    transparent or textured plastic bottles.
    """
    score = (texture * 0.70) - (gloss * 0.50) + (darkness * texture * 0.40)
    return float(np.clip(score, 0.0, 1.0))

OOD_THRESHOLD = 0.70   # tunable — raise to be less aggressive


# ══════════════════════════════════════════════════════════════════════════════
# IDEA 3 — Sensor Fusion Routing Engine
# ══════════════════════════════════════════════════════════════════════════════

def determine_sensor_route(cls_name: str, conf: float, efp: float, dynamic_thresh: float, is_anomaly: bool) -> str:
    """
    Determines which sensor path this object takes through the pipeline.
    Logs escalation decisions for the ACE rolling-window.
    """
    global _recent_escalations

    if is_anomaly:
        _recent_escalations.append(0)
        return "REJECT (OOD Hazard)"

    if cls_name not in ("PLASTIC", "TRASH"):
        # Non-plastic: visual classification is sufficient
        _recent_escalations.append(0)
        return "RGB Only"

    # PLASTIC routing
    if efp > 0.60:
        # NIR will fail → skip directly to MIR (Idea 2 bypass)
        _recent_escalations.append(1)
        return "RGB → MIR  [NIR bypassed via EFP]"

    if conf < dynamic_thresh:
        # Low confidence even after dynamic gate → escalate RGB → NIR → MIR
        _recent_escalations.append(1)
        return "RGB → NIR → MIR  [low conf]"

    # Confident enough for NIR only
    _recent_escalations.append(0)
    return "RGB → NIR"


# ══════════════════════════════════════════════════════════════════════════════
# GUI DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def draw_sorting_panel(frame_h: int, detections: list, dynamic_thresh: float, ambient_lux: float) -> np.ndarray:
    W = 460
    panel = np.zeros((frame_h, W, 3), dtype=np.uint8)

    # Gradient background
    for y in range(frame_h):
        v = int(18 + (y / frame_h) * 12)
        panel[y] = (v, v, v + 4)

    # Header
    cv2.rectangle(panel, (0, 0), (W, 64), (22, 22, 28), -1)
    cv2.putText(panel, "UNIFIED MULTIMODAL PIPELINE",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (90, 195, 255), 2)
    ace_label = "ACE: ACTIVE" if ACE_ENGINE else "ACE: fallback"
    efp_label = "EFP: ACTIVE" if EFP_MODEL else "EFP: fallback"
    cv2.putText(panel, f"Gate={dynamic_thresh:.0%}  Lux={ambient_lux:.0f}  {ace_label}  {efp_label}",
                (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 90), 1)
    cv2.line(panel, (8, 64), (W - 8, 64), (60, 60, 60), 1)

    y_off = 78
    CARD_H = 128
    for det in detections[:3]:
        cls_name   = det["class"]
        conf       = det["confidence"]
        efp        = det["efp"]
        route      = det["route"]
        is_anomaly = det["anomaly"]
        anom_score = det["anomaly_score"]

        key   = "ANOMALY" if is_anomaly else cls_name
        style = SORTING_MAP.get(key, SORTING_MAP["ANOMALY"])
        c     = style["color"]
        bg    = style["bg"]

        # Card
        cv2.rectangle(panel, (8, y_off), (W - 8, y_off + CARD_H), bg, -1)
        cv2.rectangle(panel, (8, y_off), (W - 8, y_off + CARD_H), c, 1)
        cv2.rectangle(panel, (8, y_off), (14, y_off + CARD_H), c, -1)  # accent bar

        disp_name = "⚠ ANOMALY" if is_anomaly else cls_name
        cv2.putText(panel, disp_name,
                    (20, y_off + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(panel, f"YOLO conf: {conf:.1%}  gate: {det['threshold']:.1%}",
                    (20, y_off + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1)

        efp_col = (40, 40, 255) if efp > 0.60 else (40, 220, 40)
        cv2.putText(panel, f"EFP: {efp:.1%}  OOD: {anom_score:.2f}",
                    (20, y_off + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, efp_col, 1)

        cv2.putText(panel, route,
                    (20, y_off + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 230, 230), 1)

        cv2.putText(panel, style["bin"],
                    (20, y_off + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.50, c, 2)

        y_off += CARD_H + 8

    if not detections:
        cv2.putText(panel, "No objects detected", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)

    return panel


def annotate_frame(frame: np.ndarray, results, base_conf: float,
                   dynamic_thresh: float, ambient_lux: float) -> tuple[np.ndarray, list]:
    detections = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            conf    = float(box.conf[0])
            cls_id  = int(box.cls[0])
            cls_name = result.names[cls_id].upper()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Visual feature extraction (deterministic)
            crop = frame[max(0, y1):min(frame.shape[0], y2),
                         max(0, x1):min(frame.shape[1], x2)]
            darkness, gloss, texture, hue, val = extract_visual_features(crop)
            obj_size_px = float((x2 - x1) * (y2 - y1))

            # OOD Anomaly Detection (deterministic, no random noise)
            anom_score  = compute_anomaly_score(darkness, texture, gloss)
            is_anomaly  = anom_score > OOD_THRESHOLD

            # EFP Prediction (Idea 2)
            efp = compute_efp(darkness, gloss, texture, obj_size_px, ambient_lux)

            # Dynamic Threshold (Idea 1 — real ACE engine)
            dyn_thresh = get_dynamic_threshold(
                base_conf, darkness, gloss, texture, hue, val,
                obj_size_px, ambient_lux, cls_name
            )

            # Sensor Route (Idea 3)
            route = determine_sensor_route(cls_name, conf, efp, dyn_thresh, is_anomaly)

            key   = "ANOMALY" if is_anomaly else cls_name
            color = SORTING_MAP.get(key, SORTING_MAP["ANOMALY"])["color"]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{'ANOMALY' if is_anomaly else cls_name} | {route}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

            detections.append({
                "class":       cls_name,
                "confidence":  conf,
                "threshold":   dyn_thresh,
                "efp":         efp,
                "route":       route,
                "anomaly":     is_anomaly,
                "anomaly_score": anom_score,
            })

    return frame, detections


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def find_best_weights() -> str | None:
    candidates = [
        PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt",
        PROJECT_ROOT / "models" / "best.pt",
        PROJECT_ROOT / "best.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def run_detection(args):
    print("\n  Multimodal Waste Segregation — Grand Unified Architecture")
    print("  " + "=" * 58)

    weights = args.weights or find_best_weights()
    if not weights:
        print("  [ERROR] No trained YOLO model found. Train the model first.")
        sys.exit(1)

    model = YOLO(weights)
    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"  [ERROR] Cannot open source: {args.source}")
        sys.exit(1)

    base_conf = args.conf
    print(f"  [✓] Camera ready. Base conf={base_conf:.0%}  Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Estimate ambient lux from the actual frame (deterministic)
        ambient_lux = compute_ambient_lux(frame)

        # Compute a single scene-level dynamic threshold for the panel header
        scene_thresh = get_dynamic_threshold(
            base_conf, 0.5, 0.5, 0.5, 0.5, 0.5, 10000.0, ambient_lux, "PLASTIC"
        )

        results = model.predict(frame, conf=0.15, iou=0.45, verbose=False)
        frame, detections = annotate_frame(frame, results, base_conf, scene_thresh, ambient_lux)
        panel  = draw_sorting_panel(frame.shape[0], detections, scene_thresh, ambient_lux)
        output = np.hstack([frame, panel])

        cv2.imshow("Unified Pipeline — Press Q to quit", output)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Unified Multimodal Waste Sorting — Real-Time")
    ap.add_argument("--weights", type=str, default=None,  help="Path to YOLO weights (.pt)")
    ap.add_argument("--source",  type=str, default="0",   help="Camera index or video path")
    ap.add_argument("--conf",    type=float, default=0.70, help="Base YOLO confidence threshold")
    run_detection(ap.parse_args())
