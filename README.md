# ♻️ AI + Sensor Based Plastic Waste Sorting System
### *Featuring the Adaptive Confidence Engine (ACE)*

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/release/python-3140/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multimodal waste segregation system integrating deep learning computer vision with spectral sensing (NIR + MIR). The system introduces a novel **Adaptive Confidence Engine (ACE)** for intelligent sensor cascade management, significantly reducing energy consumption while maintaining peak sorting accuracy.

---

## 🌟 Key Highlights

- **🎯 Macro-Class Visual Sorting** across 6 categories (`BIODEGRADABLE`, `CARDBOARD`, `GLASS`, `METAL`, `PAPER`, `PLASTIC`) using YOLOv11s.
- **🔍 97.12% NIR Spectral Identification** accuracy for primary polymer typing using SVM on 232-band spectra.
- **🖤 100% MIR Identification Accuracy** on pure spectral samples (`PET`, `HDPE`, `PVC`, `LDPE`, `PP`, `PS`) using Random Forest / SVM classifiers.
- **📈 Adaptive Confidence Engine (ACE)** trained on 17 features achieving 99.8% threshold prediction accuracy.
- **🔋 Proactive Expected Failure Prediction (EFP)** model (ROC-AUC 1.00) bypassing 13.8% of items (seeded by mean reflectance < 0.08 black plastic gate) straight to MIR.

---

## 🏗️ System Architecture

The project features a highly optimized, cascaded pipeline combining visual and spectral modalities. 

```mermaid
graph TD
    %% Define Styles
    classDef vision fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px;
    classDef spectral fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
    classDef decision fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    classDef action fill:#fce4ec,stroke:#e91e63,stroke-width:2px;
    classDef db fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px;

    %% Nodes
    In[📸 Incoming Conveyor Belt Image] --> S1
    
    subgraph Vision Pipeline
        S1[Stage 1: YOLOv11s Object Detection<br/>6 Macro-classes]:::vision
    end

    S1 -- "Non-Plastic" --> B1(♻️ Diversion Bins: Green/Blue/Yellow Bins)
    S1 -- "PLASTIC" --> EFP

    subgraph Spectral Sensing & ACE
        EFP{EFP Predictor<br/>NIR fail / black plastic?}:::decision
        EFP -- "Yes (mean_refl < 0.08)" --> S5
        EFP -- "No" --> S3
        
        S3[Stage 2: NIR Spectral Sensing<br/>SVM on 232-band spectra]:::spectral
        S3 --> ACE
        
        ACE{Stage 3: Adaptive Confidence Engine<br/>XGBoost threshold prediction}:::decision
        ACE -- "Confidence ≥ Threshold" --> Out[✅ Final Polymer Classification]
        ACE -- "Confidence < Threshold" --> S5
        
        S5[Stage 4: MIR Fallback Sensing<br/>Random Forest on FTIR spectra]:::spectral
        S5 --> Out
    end
    
    Out --> Arm[🤖 Robotic Arm Sorting]:::action
```

### Stage Breakdown
1. **Stage 1 — Vision Detection:** YOLOv11s detects waste objects on the conveyor belt and classifies them into 6 macro-classes.
2. **Expected Failure Predictor (EFP):** Ingests visual properties (e.g. darkness, gloss) and fast preliminary NIR mean intensity. If it predicts an NIR failure (e.g., carbon-black absorbing plastics, where mean reflectance < 0.08), it bypasses the NIR stage entirely.
3. **Stage 2 — NIR Spectral Sensing:** Employs an SVM classifier on 232-band NIR spectra for primary polymer typing.
4. **Stage 3 — Adaptive Confidence Engine (ACE):** An XGBoost model dynamically determines if the costly MIR sensor is required based on a 17-feature context vector.
5. **Stage 4 — MIR Fallback:** High-precision FTIR spectra analyzed by a Random Forest model, specifically targeted at black or heavily contaminated plastics (100% accuracy on pure samples).

---

## 📁 Project Structure

```text
├── src/
│   ├── ace/                          # Adaptive Confidence Engine
│   │   ├── engine.py                 # Core ACE (XGBoost wrapper)
│   │   ├── feature_extractor.py      # 17-feature context vector extractor
│   │   ├── synthetic_data.py         # Physics-based data generator
│   │   ├── feedback_db.py            # SQLite feedback loop
│   │   ├── train_ace.py              # Train ACE on synthetic data
│   │   └── evaluate_ace.py           # ACE vs Fixed Threshold comparison
│   ├── detect.py                     # YOLOv11s single-frame webcam inference
│   └── unified_pipeline.py           # Grand Unified sorting pipeline with real MIR models
├── Atharva_NIR/                      # Spectral Sensing Track
│   ├── preprocess.py                 # SNV Normalization + Savgol filtering
│   ├── train_classical.py            # MIR classifier training (RF, SVM, XGBoost)
│   ├── results/                      # Validation logs & paper metrics
│   └── models/                       # Pre-trained MIR/NIR pickle models & scalers
├── docs/                             # Documentation and Reports
│   ├── PAPER_ALIGNMENT.md            # Patent/Paper vs Code verification tracker
│   └── efp_report.md                 # Expected Failure Predictor training report
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.14+
- `scikit-learn`, `xgboost`, `opencv-python`, `scipy`, `pandas`, `joblib`

### Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows

# Install Dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run the Unified Segregation Pipeline
To run the end-to-end multimodal cascade demo using real MIR spectra from validation splits:
```bash
python -m src.unified_pipeline
```

### 2. Retrain EFP Model
```bash
python scripts/train_efp_model.py
```

### 3. Retrain ACE Engine
```bash
python -m src.ace.train_ace
```

---

## 📖 Citation

If you use this code or the ACE methodology in your research, please cite our IEEE paper draft:

```bibtex
@inproceedings{author2026ace,
  title={Adaptive Confidence Engine for Energy-Efficient Multimodal Plastic Waste Sorting},
  author={Placeholder, Author},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026},
  organization={IEEE}
}
```
