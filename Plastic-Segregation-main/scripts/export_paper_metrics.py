"""Export committed spectral metrics into a paper-ready markdown table."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NIR = ROOT / "Atharva_NIR" / "results"
OUT = ROOT / "results" / "vision"
OUT_SPECTRAL = ROOT / "results" / "spectral"


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main():
    OUT_SPECTRAL.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    lines = ["# Spectral metrics (from committed Atharva_NIR/results)", ""]

    # NIR
    nir = load_json(NIR / "nir_hsi_results.json")
    lines.append("## NIR-HSI classical models")
    lines.append("| Model | Test Acc | Precision | Recall | F1 |")
    lines.append("|-------|----------|-----------|--------|----|")
    for name, m in nir.items():
        lines.append(
            f"| {name} | {m['test_accuracy']:.4f} | {m['test_precision']:.4f} | "
            f"{m['test_recall']:.4f} | {m['test_f1']:.4f} |"
        )
    lines.append("")

    # MIR comparison CSV
    csv_path = NIR / "model_comparison.csv"
    lines.append("## MIR FTIR model comparison")
    lines.append("| Model | Test Acc | Precision | Recall | F1 |")
    lines.append("|-------|----------|-----------|--------|----|")
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # first column may be unnamed index
        for row in reader:
            model = row.get("") or row.get("Model") or next(iter(row.values()))
            # DictReader with leading comma uses '' as key for index col
            keys = list(row.keys())
            name = row[keys[0]]
            lines.append(
                f"| {name} | {float(row['test_accuracy']):.4f} | "
                f"{float(row['test_precision']):.4f} | "
                f"{float(row['test_recall']):.4f} | {float(row['test_f1']):.4f} |"
            )
    lines.append("")

    best = load_json(NIR / "best_model_info.json")
    lines.append("## Best MIR model")
    lines.append(f"- Name: **{best.get('best_model_name')}**")
    lines.append(f"- Test accuracy: **{best.get('test_accuracy')}**")
    lines.append(f"- Test F1: **{best.get('test_f1')}**")
    lines.append("")
    lines.append("## Vision metrics")
    lines.append(
        "_Not committed yet. Run `python src/validate.py --weights models/best.pt` "
        "and save output under `results/vision/`. Place screenshots from "
        "`python src/detect.py` alongside._"
    )

    out_md = OUT_SPECTRAL / "paper_metrics.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")

    # also copy best info json for convenience
    (OUT_SPECTRAL / "best_model_info.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
