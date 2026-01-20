# ChronoFace - Age Prediction Module (Part 1)

**Author:** David Ceylon  
**Project:** ChronoFace - Time-based Facial Analysis & Synthesis  
**Core Tech:** Swin Transformer (PyTorch/Timm)

---

## 📋 Overview
This module focuses on the **biometric estimation of biological age** from facial images. Unlike traditional approaches using CNNs (like VGG or ResNet), this project leverages a **Vision Transformer (Swin Transformer)** to capture global dependencies in facial structures (shape, skin texture, wrinkles) for higher accuracy.

The model treats age prediction as a classification problem (101 classes, ages 0-100) and computes the final age using the **Expected Value (DEX approach)**, achieving a Mean Absolute Error (MAE) of **6.0 years**.

### Key Features
* **Swin Transformer Backbone:** Uses `swin_base_patch4_window7_224` (via `timm` library).
* **Transfer Learning:** Pretrained on ImageNet-22k and fine-tuned on UTKFace.
* **Robust Preprocessing:** Auto-cleaning of corrupted images and stratified data splitting.
* **DEX Output:** Uses Softmax probability distribution to calculate the weighted average age.

---

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* PyTorch (CUDA support highly recommended)
* Google Colab (Premium recommended for VRAM)

### Dependencies
Install the required libraries using pip:

```bash
pip install torch torchvision timm pandas numpy scikit-learn tqdm pillow
