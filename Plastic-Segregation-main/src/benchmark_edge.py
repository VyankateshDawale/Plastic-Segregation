"""
benchmark_edge.py — Edge deployment benchmarking
================================================
Measures inference performance of the system's models on the dev machine
(NVIDIA RTX 4050 Laptop GPU, 6 GB) and exports them for edge deployment:

    - FPS / latency per stage (detection, classification, non-plastic cls)
    - Peak GPU memory per model
    - ONNX export  (universal, CPU/GPU edge targets)
    - TensorRT export (Jetson / Orin) — requires the `tensorrt` package

The script is safe to run even if some models are absent or some export
backends are missing: it benchmarks whatever exists and reports NA for the
rest, so it always produces a usable benchmark CSV for the paper.

Jetson / Raspberry Pi numbers require physical hardware — see the plan's
"Needs External Resources" section. This script reports the RTX 4050 laptop
numbers and notes that Jetson deployment is planned future work.

Usage:
    python src/benchmark_edge.py
    python src/benchmark_edge.py --runs 100
    python src/benchmark_edge.py --skip-export
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent

MODELS = {
    "detection": PROJECT_ROOT / "runs" / "detect_v2" / "train" / "weights" / "best.pt",
    "classification": PROJECT_ROOT / "runs" / "classify" / "train" / "weights" / "best.pt",
    "nonplastic_cls": PROJECT_ROOT / "runs" / "nonplastic_cls" / "train" / "weights" / "best.pt",
}
# Fallback to V1 detection if V2 not trained yet
OLD_DETECT = PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "train" / "weights" / "best.pt"


def gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return float("nan")


def benchmark_model(model_path: Path, dummy, runs: int, warmup: int = 5):
    """Measure median latency + FPS over `runs` inferences. Returns dict."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))

    # warmup
    for _ in range(warmup):
        _ = model(dummy, verbose=False)

    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(dummy, verbose=False)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    lat = np.array(latencies) * 1000.0  # ms
    peak_mb = gpu_memory_mb()
    return {
        "median_latency_ms": float(np.median(lat)),
        "mean_latency_ms": float(np.mean(lat)),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "fps": float(1000.0 / np.median(lat)),
        "peak_vram_mb": round(peak_mb, 1),
    }


def try_export_onnx(model_path: Path, out_dir: Path):
    from ultralytics import YOLO
    try:
        m = YOLO(str(model_path))
        exported = m.export(format="onnx", imgsz=640, half=False,
                            int8=False, dynamic=False, workspace=4)
        return str(exported)
    except Exception as e:  # noqa: BLE001
        return f"FAILED: {type(e).__name__}: {e}"


def try_export_trt(model_path: Path, out_dir: Path):
    from ultralytics import YOLO
    try:
        import tensorrt  # noqa: F401  (only needed to confirm availability)
        m = YOLO(str(model_path))
        exported = m.export(format="engine", imgsz=640, half=True,
                            int8=False, dynamic=False, workspace=4)
        return str(exported)
    except ImportError:
        return "SKIPPED: tensorrt package not installed (Jetson build env required)"
    except Exception as e:  # noqa: BLE001
        return f"FAILED: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "benchmark"))
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = out_dir / "exports"
    export_dir.mkdir(exist_ok=True)

    device = "0" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("⚡ EDGE BENCHMARK — RTX 4050 Laptop (6 GB)")
    print("=" * 60)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device: {gpu_name}  | runs={args.runs} | dummy imgsz={args.imgsz}")

    # Resolve detection model (prefer V2, fallback V1)
    det_model = MODELS["detection"]
    if not det_model.exists() and OLD_DETECT.exists():
        print("  ⚠️  V2 detect model missing; using V1 for benchmark.")
        det_model = OLD_DETECT

    dummy_rgb = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    rows = []
    for name, path in MODELS.items():
        if name == "detection":
            path = det_model
        if not path.exists():
            print(f"  ⚠️  {name}: model not found ({path}) — skipped")
            rows.append({
                "model": name, "median_latency_ms": "NA", "mean_latency_ms": "NA",
                "p95_latency_ms": "NA", "fps": "NA", "peak_vram_mb": "NA",
                "onnx": "NA", "tensorrt": "NA",
            })
            continue
        print(f"\n  🔧 Benchmarking {name} ...")
        try:
            bench = benchmark_model(path, dummy_rgb, args.runs)
        except Exception as e:  # noqa: BLE001
            print(f"     ❌ benchmark failed: {e}")
            rows.append({"model": name, "median_latency_ms": "ERR",
                         "mean_latency_ms": "ERR", "p95_latency_ms": "ERR",
                         "fps": "ERR", "peak_vram_mb": "ERR",
                         "onnx": "NA", "tensorrt": "NA"})
            continue

        onnx = try_export_onnx(path, export_dir) if not args.skip_export else "SKIPPED (--skip-export)"
        trt = try_export_trt(path, export_dir) if not args.skip_export else "SKIPPED (--skip-export)"

        print(f"     median {bench['median_latency_ms']:.1f} ms  "
              f"FPS {bench['fps']:.1f}  VRAM {bench['peak_vram_mb']} MB")
        rows.append({
            "model": name,
            "median_latency_ms": round(bench["median_latency_ms"], 2),
            "mean_latency_ms": round(bench["mean_latency_ms"], 2),
            "p95_latency_ms": round(bench["p95_latency_ms"], 2),
            "fps": round(bench["fps"], 2),
            "peak_vram_mb": bench["peak_vram_mb"],
            "onnx": (onnx if isinstance(onnx, str) else str(onnx)),
            "tensorrt": (trt if isinstance(trt, str) else str(trt)),
        })

    # End-to-end latency estimate (detection -> classification per crop)
    det_row = next((r for r in rows if r["model"] == "detection"), None)
    cls_row = next((r for r in rows if r["model"] == "classification"), None)
    if det_row and cls_row and det_row["fps"] not in ("NA", "ERR"):
        e2e_ms = det_row["median_latency_ms"] + cls_row["median_latency_ms"]
        print(f"\n  🔗 Est. per-object pipeline latency (detect+classify): {e2e_ms:.1f} ms "
              f"(~{1000.0 / e2e_ms:.1f} obj/s theoretical)")

    # Write CSV
    csv_path = out_dir / "edge_benchmark.csv"
    fieldnames = ["model", "median_latency_ms", "mean_latency_ms", "p95_latency_ms",
                  "fps", "peak_vram_mb", "onnx", "tensorrt"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n💾 Benchmark CSV: {csv_path}")
    print("   Jetson/RPi numbers require physical hardware (planned future work).")


if __name__ == "__main__":
    main()
