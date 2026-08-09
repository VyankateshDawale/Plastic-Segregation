"""Train the Adaptive Confidence Engine on synthetic data.

Usage:
    python src/ace/train_ace.py
    python src/ace/train_ace.py --samples 20000 --output models/ace_xgboost.json
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ace.synthetic_data import SyntheticDataGenerator
from src.ace.feature_extractor import FeatureExtractor
from src.ace.engine import ACEEngine


def main():
    parser = argparse.ArgumentParser(description="Train the Adaptive Confidence Engine")
    parser.add_argument("--samples", type=int, default=10000, help="Number of synthetic samples")
    parser.add_argument("--output", type=str, default="models/ace_xgboost.json", help="Model save path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs("outputs/ace", exist_ok=True)

    # 1. Generate synthetic data
    print(f"Generating {args.samples} synthetic samples...")
    generator = SyntheticDataGenerator(seed=42)
    X, y = generator.generate(n_samples=args.samples)

    # 2. Dataset statistics
    feature_names = FeatureExtractor().feature_names()
    print(f"\n{'='*50}")
    print(f"  DATASET STATISTICS")
    print(f"{'='*50}")
    print(f"  Total samples:      {len(y)}")
    print(f"  MIR needed (pos):   {int(y.sum())} ({y.mean():.1%})")
    print(f"  NIR sufficient:     {int((1-y).sum())} ({(1-y).mean():.1%})")
    print(f"  Features:           {X.shape[1]}")

    # 3. Train ACEEngine
    print(f"\n{'='*50}")
    print(f"  TRAINING ACE ENGINE (XGBoost)")
    print(f"{'='*50}")
    engine = ACEEngine()
    engine.feature_names = feature_names
    metrics = engine.fit(X, y, val_split=0.2)

    # 4. Print detailed metrics
    print(f"\n{'='*50}")
    print(f"  EVALUATION METRICS")
    print(f"{'='*50}")
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.4f}")

    # 5. Feature importance
    importance = engine.feature_importance()
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'='*50}")
    print(f"  FEATURE IMPORTANCE RANKING")
    print(f"{'='*50}")
    for rank, (feat, imp) in enumerate(sorted_imp, 1):
        bar = '=' * int(imp * 100)
        print(f"  {rank:2d}. {feat:28s} {imp:.4f}  {bar}")

    # 6. Save model
    engine.save(args.output)
    print(f"\n[OK] Model saved to: {args.output}")

    # 7. Feature importance plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 7))
    names = [x[0] for x in sorted_imp][::-1]
    values = [x[1] for x in sorted_imp][::-1]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    ax.barh(names, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title('Adaptive Confidence Engine — Feature Importance', fontsize=14, fontweight='bold')
    ax.set_xlabel('XGBoost Importance Score', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('outputs/ace/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Feature importance plot saved to: outputs/ace/feature_importance.png")

    # Quick demo: show adaptive thresholds for a few samples
    print(f"\n{'='*50}")
    print(f"  SAMPLE ADAPTIVE THRESHOLDS")
    print(f"{'='*50}")
    for i in [0, 1, 2, len(X)//2, len(X)-1]:
        threshold = engine.predict_threshold(X[i])
        nir_conf = X[i, 0]
        escalate, thresh = engine.should_escalate(nir_conf, X[i])
        label = "-> ESCALATE to MIR" if escalate else "-> Accept NIR"
        print(f"  Sample {i:5d}: NIR conf={nir_conf:.3f}, ACE threshold={thresh:.3f} {label}")

if __name__ == "__main__":
    main()
