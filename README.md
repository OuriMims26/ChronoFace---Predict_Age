# ChronoFace - Age Prediction Module (Part 1) ⏳

**Authors:** Ouriel Mimoun & David Ceylon  
**Project:** ChronoFace - Time-based Facial Analysis & Synthesis  
**Core Tech:** Swin Transformer (PyTorch/Timm)

---

## 📋 Overview
This module focuses on the **biometric estimation of biological age** from facial images. Unlike traditional approaches using CNNs (like VGG or ResNet), this project leverages a **Vision Transformer (Swin Transformer)** to capture global dependencies in facial structures (shape, skin texture, wrinkles) for higher accuracy.

The model treats age prediction as a classification problem (121 classes, ages 0-120) and computes the final age using the **Expected Value (DEX approach)**, achieving high precision by using the Softmax probability distribution to calculate the weighted average age.

### Key Features
* **Swin Transformer Backbone:** Uses `swin_base_patch4_window7_224` (via `timm` library).
* **Transfer Learning:** Pretrained on ImageNet and fine-tuned on specialized aging datasets.
* **DEX Output:** Uses the **Deep EXpectation** method where: `Age = Σ (Probability_i * i)` for i in [0, 120].
* **Robust Preprocessing:** Automated resizing (224x224) and ImageNet normalization.

---

## 🧠 Model Architecture
The architecture consists of:
- **Backbone**: Swin Transformer Base.
- **Custom Head**: 
  - Linear Layer (1024 -> 512)
  - ReLU Activation
  - Dropout (0.5)
  - Final classification layer (512 -> 121 classes).

---

## 📂 Required Files
To run this module, the following weight file must be in the project root:
- **`checkpoint_fold_3.pth`**: Trained weights for the Swin Transformer model.

📥 **Download all weights here**: [Google Drive - ChronoFace Weights](https://drive.google.com/drive/folders/10_EO8AE7Fqk7aJcVDLWeXDiSA9fClN15?usp=sharing) (Please place them in the project root folder).

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- A virtual environment (recommended)
- PyTorch (MPS support for Mac or CUDA for NVIDIA)

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies: `torch`, `torchvision`, `timm`, `streamlit`, `pillow`, `matplotlib`, `numpy`, `pillow-heif`)*

---

## 🚀 How to Run
The Age Prediction module is integrated into the ChronoFace dashboard:

```bash
streamlit run app.py
```

1. Upload a portrait image.
2. Click on **"1. Predict Age 🚀"**.
3. View the estimated age and the probability distribution chart.

---

## 📊 Technical Specifications
- **Input Resolution**: 224x224 pixels.
- **Normalization**: Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`.
- **Device Support**: Automatically toggles between **MPS** (Metal), **CUDA**, or **CPU**.
