# 🗑️ Plastic Waste Sorting System

Real-time waste detection and sorting recommendation system using **YOLOv11**, **OpenCV**, and a laptop webcam.

## 🎯 Features

- **Real-time Detection**: Live webcam feed with object detection
- **6 Waste Classes**: Biodegradable, Cardboard, Glass, Metal, Paper, Plastic
- **Sorting Recommendations**: Automatic bin assignment (Green/Blue/Yellow)
- **Confidence Scoring**: Visual confidence bars for each detection
- **Session Statistics**: Track detection counts over time
- **Interactive Controls**: Pause, screenshot, adjust confidence threshold

## 🏗️ Architecture

```
Laptop Webcam → OpenCV → YOLOv11s → Waste Dataset → Real-time Detection → Sorting Panel
```

## 📁 Project Structure

```
├── dataset/                    # Downloaded dataset (you add this)
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
├── src/
│   ├── train.py               # Model training
│   ├── detect.py              # Real-time webcam detection
│   ├── validate.py            # Model validation on test set
│   └── sanity_check.py        # Dataset verification
├── runs/                      # Training results (auto-generated)
├── outputs/                   # Screenshots from detection
├── venv/                      # Python virtual environment
└── requirements.txt
```

## 🚀 Quick Start

### 1. Setup (already done if you followed the guide)

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies (PyTorch with CUDA for RTX 4050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the [Garbage Detection Dataset](https://www.kaggle.com/datasets/sshikamaru/garbage-detection) from Kaggle and extract it into the `dataset/` folder.

### 3. Verify Dataset

```bash
python src/sanity_check.py
```

### 4. Train Model

```bash
python src/train.py                     # Train with defaults (100 epochs)
python src/train.py --epochs 150        # More epochs
python src/train.py --batch 8           # Lower batch size if OOM
python src/train.py --resume            # Resume interrupted training
```

### 5. Validate

```bash
python src/validate.py
```

### 6. Run Detection

```bash
python src/detect.py                    # Default webcam
python src/detect.py --source 1         # External webcam
python src/detect.py --conf 0.5         # Higher confidence threshold
```

## 🎮 Detection Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `S` | Save screenshot |
| `P` | Pause / Resume |
| `+` / `-` | Increase / Decrease confidence |
| `R` | Reset statistics |

## 🗂️ Sorting Bins

| Waste Type | Bin Color | Instructions |
|------------|-----------|--------------|
| Biodegradable | 🟢 Green | Compost / Organic waste |
| Cardboard | 🔵 Blue | Flatten before disposal |
| Glass | 🟡 Yellow | Handle with care, rinse first |
| Metal | 🟡 Yellow | Crush if possible |
| Paper | 🔵 Blue | Keep dry |
| Plastic | 🟡 Yellow | Check resin code |

## ⚙️ Tech Stack

- **Python 3.10+**
- **YOLOv11s** (Ultralytics) — Object detection
- **PyTorch + CUDA 12.6** — GPU-accelerated training
- **OpenCV** — Webcam capture & visualization
- **RTX 4050** — Local GPU training

## 📄 License

Dataset: CC BY 4.0 | Code: MIT
