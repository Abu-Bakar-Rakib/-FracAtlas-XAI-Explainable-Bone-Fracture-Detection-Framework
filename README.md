# 🏥 FracAtlas-XAI: Explainable Bone Fracture Detection Framework

> An advanced deep learning framework for automated bone fracture detection and classification with explainable AI integration for clinical decision support.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Medical Imaging](https://img.shields.io/badge/Domain-Medical%20Imaging-brightgreen)]()


## 🎯 Overview

FracAtlas-XAI is a state-of-the-art framework for detecting and classifying bone fractures in radiographic images. The system employs a two-stage deep learning pipeline combined with explainable AI (XAI) techniques to provide transparent, clinically interpretable results for radiologists and medical professionals.

**Key Innovation:** Integration of CycleGAN-based synthetic data generation to handle class imbalance and multiple explainability methods (Grad-CAM, Grad-CAM++, Eigen-CAM) for trustworthy AI in medical diagnosis.

## 🔥 Key Features

### ✅ Two-Stage Deep Learning Pipeline
- **Stage 1:** Binary fracture detection with high sensitivity
- **Stage 2:** Bone region classification for anatomical localization

### 🧠 State-of-the-Art Models
- **DenseNet121** for classification tasks with optimal feature extraction
- **YOLOv10m** for real-time fracture localization and bounding box regression

### 🎯 High Performance Metrics
| Metric | Value | Task |
|--------|-------|------|
| Accuracy | 91.84% | Fracture Detection |
| Accuracy | 97.31% | Bone Classification |
| mAP@50 | 0.995 | Fracture Localization |
| Inference Speed | ~50ms | Per Image (GPU) |

### ⚖️ Smart Class Imbalance Handling
- CycleGAN-based synthetic data generation
- Automatic data augmentation pipeline
- Balanced dataset curation

### 🔍 Explainable AI Integration
- **Grad-CAM:** Gradient-based visualization
- **Grad-CAM++:** Weighted gradient visualization
- **Eigen-CAM:** Eigen-value based attention maps
- Attention mechanism visualization
- Feature importance analysis

## 📊 Architecture

```
Input Image (X-ray)
        ↓
┌───────────────────────┐
│   Stage 1: Detection  │ (DenseNet121)
│   Fracture/No-Fracture│
└───────────────────────┘
        ↓
┌───────────────────────┐
│  Stage 2: Localization│ (YOLOv10m)
│   Bounding Box + Class│
└───────────────────────┘
        ↓
┌───────────────────────┐
│  XAI Component        │
│    Grad-CAM           │
│    Grad-CAM++         │  
│     EigenCAM          │
│     Analysis          │
└───────────────────────┘
        ↓
 Visualization
```

## 💻 Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM (16GB recommended)

### Clone Repository
```bash
git clone https://github.com/Abu-Bakar-Rakib/-FracAtlas-XAI-Explainable-Bone-Fracture-Detection-Framework.git
cd -FracAtlas-XAI-Explainable-Bone-Fracture-Detection-Framework
```
## 🔍 Explainability

### Grad-CAM
Visualizes the regions of the input image that are most important for the neural network's prediction.

### Grad-CAM++
An improved version of Grad-CAM providing better localization, especially for multi-object scenarios.

### Eigen-CAM
Uses the principal components of the feature maps to generate more stable attention visualizations.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support & Contact

- **Issues:** [GitHub Issues](https://github.com/Abu-Bakar-Rakib/-FracAtlas-XAI-Explainable-Bone-Fracture-Detection-Framework/issues)
- **Email:** rakibcdp@gmail.com

--- 
**Maintainer:** Abu-Bakar Rakib  
