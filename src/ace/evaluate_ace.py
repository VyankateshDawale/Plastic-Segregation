"""Evaluate ACE vs Fixed Threshold — generates publication-quality comparison.

Usage:
    python src/ace/evaluate_ace.py
    python src/ace/evaluate_ace.py --test-samples 5000
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from src.ace.synthetic_data import SyntheticDataGenerator
from src.ace.feature_extractor import FeatureExtractor
from src.ace.engine import ACEEngine


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACE vs Fixed Threshold")
    parser.add_argument("--test-samples", type=int, default=5000, help="Number of test samples")
    args = parser.parse_args()

    os.makedirs("outputs/ace", exist_ok=True)
    model_path = "models/ace_xgboost.json"

    # 1. Load trained ACE model
    print(f"Loading trained ACE model from {model_path}...")
    engine = ACEEngine()
    engine.load(model_path)
    feature_names = FeatureExtractor().feature_names()
    engine.feature_names = feature_names

    # 2. Generate SEPARATE test dataset (different seed!)
    print(f"Generating {args.test_samples} test samples (seed=99)...")
    generator = SyntheticDataGenerator(seed=99)
    X_test, y_test = generator.generate(n_samples=args.test_samples)
    n = len(y_test)

    nir_confidence = X_test[:, 0]       # feature 0 = nir_confidence
    reflectance_mean = X_test[:, 2]     # feature 2 = reflectance_mean
    mir_needed = y_test.astype(bool)    # ground truth: was MIR actually needed?
    nir_correct = ~mir_needed           # NIR was sufficient

    # 3. Compute decisions for BOTH strategies
    FIXED_THRESHOLD = 0.85
    fixed_escalate = nir_confidence < FIXED_THRESHOLD

    ace_escalate = np.zeros(n, dtype=bool)
    ace_thresholds = np.zeros(n)
    ace_probs = np.zeros(n)
    for i in range(n):
        escalate, threshold = engine.should_escalate(nir_confidence[i], X_test[i])
        ace_escalate[i] = escalate
        ace_thresholds[i] = threshold
        # Get raw probability for ROC curve
        if X_test[i].ndim == 1:
            feats = X_test[i].reshape(1, -1)
        else:
            feats = X_test[i]
        ace_probs[i] = engine.model.predict_proba(feats)[0, 1]

    # 4. Compute metrics for both strategies
    def compute_metrics(escalate_decision):
        # Correct if: (didn't escalate AND NIR was correct) OR (escalated, MIR always correct)
        correct = (~escalate_decision & nir_correct) | escalate_decision
        sorting_accuracy = correct.mean()
        mir_activation_rate = escalate_decision.mean()
        false_escalations = (escalate_decision & nir_correct).mean()
        missed_escalations = (~escalate_decision & mir_needed).mean()
        mir_count = escalate_decision.sum()
        energy_proxy = mir_count * 3.0
        # Throughput: 18 items/min baseline, each MIR adds 0.5s delay
        base_time_s = n * (60.0 / 18.0)
        total_time_s = base_time_s + (mir_count * 0.5)
        throughput = n / (total_time_s / 60.0)
        cost_per_item = energy_proxy / n

        return {
            "sorting_accuracy": sorting_accuracy,
            "mir_activation_rate": mir_activation_rate,
            "false_escalations": false_escalations,
            "missed_escalations": missed_escalations,
            "energy_proxy": energy_proxy,
            "throughput_ipm": throughput,
            "cost_per_item": cost_per_item,
        }

    fixed_metrics = compute_metrics(fixed_escalate)
    ace_metrics = compute_metrics(ace_escalate)

    # 5. Print formatted comparison table
    print(f"\n{'='*70}")
    print(f"  ACE vs FIXED THRESHOLD COMPARISON ({n} test samples)")
    print(f"{'='*70}")
    print(f"  {'Metric':<28} {'Fixed (0.85)':>14} {'ACE (Adaptive)':>14} {'Δ':>10}")
    print(f"  {'-'*66}")
    for k in fixed_metrics:
        fv = fixed_metrics[k]
        av = ace_metrics[k]
        delta = av - fv
        sign = "+" if delta > 0 else ""
        if 'accuracy' in k or 'rate' in k or 'escalation' in k:
            print(f"  {k:<28} {fv:>13.1%} {av:>13.1%} {sign}{delta:>9.1%}")
        else:
            print(f"  {k:<28} {fv:>13.2f} {av:>13.2f} {sign}{delta:>9.2f}")
    print(f"{'='*70}")

    # 6. Save results
    import csv
    with open('outputs/ace/ace_vs_fixed.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Fixed_Threshold', 'ACE_Adaptive', 'Delta'])
        for k in fixed_metrics:
            writer.writerow([k, f"{fixed_metrics[k]:.6f}", f"{ace_metrics[k]:.6f}",
                           f"{ace_metrics[k] - fixed_metrics[k]:.6f}"])
    print("✅ Results saved to: outputs/ace/ace_vs_fixed.csv")

    # 7. Generate publication-quality plots
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        plt.style.use('ggplot')

    # ── Plot A: Threshold Distribution ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ace_thresholds, bins=50, alpha=0.7, color='#00bcd4', edgecolor='white',
            linewidth=0.5, label='ACE Adaptive Thresholds')
    ax.axvline(FIXED_THRESHOLD, color='#ff5252', linestyle='--', linewidth=2.5,
               label=f'Fixed Threshold ({FIXED_THRESHOLD})')
    ax.set_title('Distribution of ACE Adaptive Thresholds vs Fixed Threshold',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Threshold Value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/ace/threshold_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot B: ROC Comparison ──
    fig, ax = plt.subplots(figsize=(8, 8))
    fpr_ace, tpr_ace, _ = roc_curve(y_test, ace_probs)
    roc_auc_ace = auc(fpr_ace, tpr_ace)
    fpr_fixed, tpr_fixed, _ = roc_curve(y_test, 1 - nir_confidence)
    roc_auc_fixed = auc(fpr_fixed, tpr_fixed)

    ax.plot(fpr_ace, tpr_ace, color='#00bcd4', linewidth=2.5,
            label=f'ACE (AUC = {roc_auc_ace:.3f})')
    ax.plot(fpr_fixed, tpr_fixed, color='#ff9800', linewidth=2.5,
            label=f'Fixed Threshold (AUC = {roc_auc_fixed:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax.set_title('ROC Curve: ACE vs Fixed Threshold', fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    plt.savefig('outputs/ace/roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot C: Metrics Comparison Bar Chart ──
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_to_plot = ['sorting_accuracy', 'mir_activation_rate', 'false_escalations', 'missed_escalations']
    labels = ['Sorting\nAccuracy ↑', 'MIR Activation\nRate ↓', 'False\nEscalations ↓', 'Missed\nEscalations ↓']
    fixed_vals = [fixed_metrics[m] for m in metrics_to_plot]
    ace_vals = [ace_metrics[m] for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35
    bars1 = ax.bar(x - width/2, fixed_vals, width, label='Fixed Threshold (0.85)',
                   color='#ff5252', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, ace_vals, width, label='ACE (Adaptive)',
                   color='#00bcd4', edgecolor='white', linewidth=0.5)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.1%}',
                ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.1%}',
                ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title('ACE vs Fixed Threshold — Key Metrics Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Rate', fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig('outputs/ace/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot D: Escalation Analysis Scatter ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Fixed threshold scatter
    ax = axes[0]
    ax.scatter(nir_confidence[~fixed_escalate & nir_correct], reflectance_mean[~fixed_escalate & nir_correct],
               c='#4caf50', alpha=0.3, s=8, label='Correct Accept')
    ax.scatter(nir_confidence[fixed_escalate & mir_needed], reflectance_mean[fixed_escalate & mir_needed],
               c='#2196f3', alpha=0.3, s=8, label='Correct Escalate')
    ax.scatter(nir_confidence[fixed_escalate & nir_correct], reflectance_mean[fixed_escalate & nir_correct],
               c='#ff9800', alpha=0.5, s=15, label='False Escalation')
    ax.scatter(nir_confidence[~fixed_escalate & mir_needed], reflectance_mean[~fixed_escalate & mir_needed],
               c='#f44336', alpha=0.7, s=20, marker='x', label='Missed Escalation')
    ax.axvline(FIXED_THRESHOLD, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_title('Fixed Threshold (0.85)', fontsize=13, fontweight='bold')
    ax.set_xlabel('NIR Confidence', fontsize=11)
    ax.set_ylabel('Reflectance Mean', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')

    # ACE scatter
    ax = axes[1]
    ax.scatter(nir_confidence[~ace_escalate & nir_correct], reflectance_mean[~ace_escalate & nir_correct],
               c='#4caf50', alpha=0.3, s=8, label='Correct Accept')
    ax.scatter(nir_confidence[ace_escalate & mir_needed], reflectance_mean[ace_escalate & mir_needed],
               c='#2196f3', alpha=0.3, s=8, label='Correct Escalate')
    ax.scatter(nir_confidence[ace_escalate & nir_correct], reflectance_mean[ace_escalate & nir_correct],
               c='#ff9800', alpha=0.5, s=15, label='False Escalation')
    ax.scatter(nir_confidence[~ace_escalate & mir_needed], reflectance_mean[~ace_escalate & mir_needed],
               c='#f44336', alpha=0.7, s=20, marker='x', label='Missed Escalation')
    ax.set_title('ACE Adaptive Threshold', fontsize=13, fontweight='bold')
    ax.set_xlabel('NIR Confidence', fontsize=11)
    ax.set_ylabel('Reflectance Mean', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')

    plt.suptitle('Escalation Decision Analysis: NIR Confidence vs Reflectance',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/ace/escalation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Plots saved to outputs/ace/:")
    print("   - threshold_distribution.png")
    print("   - roc_comparison.png")
    print("   - metrics_comparison.png")
    print("   - escalation_analysis.png")


if __name__ == "__main__":
    main()
