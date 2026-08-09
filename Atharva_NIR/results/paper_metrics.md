# Spectral metrics (from committed Atharva_NIR/results)

## NIR-HSI classical models
| Model | Test Acc | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| RandomForest | 0.9136 | 0.9137 | 0.9136 | 0.9121 |
| SVM | 0.9712 | 0.9713 | 0.9712 | 0.9711 |
| XGBoost | 0.9342 | 0.9350 | 0.9342 | 0.9339 |

## MIR FTIR model comparison
| Model | Test Acc | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| RandomForest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 0.9975 | 0.9975 | 0.9975 | 0.9975 |
| 1D_CNN | 0.9497 | 0.9524 | 0.9497 | 0.9479 |
| SpectralTransformer | 0.9271 | 0.9282 | 0.9271 | 0.9269 |

## Best MIR model
- Name: **RandomForest**
- Test accuracy: **1.0**
- Test F1: **1.0**

## Vision metrics
_Not committed yet. Run `python src/validate.py --weights models/best.pt` and save output under `results/vision/`. Place screenshots from `python src/detect.py` alongside._
