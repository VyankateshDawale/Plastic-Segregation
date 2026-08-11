# Expected Failure Probability (EFP) Model — Training Report

## Data Provenance
**SYNTHETIC DATA ONLY.** No real labeled NIR-fail/succeed events exist yet.
Metrics in this report must NOT be reported as empirically validated results
until real carbon-black sample measurements are available.

## Label Design (Non-tautological)
The NIR-failure label is derived from a noisy weighted combination of visual
inputs:

```
p_fail = 0.50 * rgb_darkness
       + 0.20 * (1 - gloss_index)
       + 0.15 * texture_roughness
       
       + noise(mu=0, sigma=0.10)
nir_failure = 1 if p_fail > 0.65 else 0
```

`nir_baseline_intensity` appears as an input feature AND contributes partially
to the label, but the dominant terms are the VISUAL features `rgb_darkness`
and `gloss_index`.  This avoids the AUC=1.00 tautology of the previous version
where the label was simply `nir_baseline_intensity < 0.08`.

## Model Inputs
| Feature | Description |
|---------|-------------|
| `rgb_darkness` | 1 - mean(gray channel) / 255, proxy for carbon-black presence |

| `lighting_lux` | Ambient illuminance (300–1000 lux) |
| `gloss_index` | Fraction of specular highlight pixels |
| `texture_roughness` | Normalised Laplacian variance |
| `object_size_cm` | Estimated object size in cm |

## Performance Metrics (Synthetic Test Set — 2,000 items)
- **Data source**: SYNTHETIC (non-tautological)
- **ROC-AUC Score**: 0.9449
- **Decision Threshold Applied**: 0.60
- **NIR Bypassed**: 0 items (0.0%)
- **Correctly skipped NIR (TP)**: 0
- **Incorrectly skipped NIR (FP)**: 0

## Status in PAPER_ALIGNMENT.md
This row is marked **Partial — synthetic labels only, not yet validated on
real NIR-fail / NIR-succeed measurements.**
