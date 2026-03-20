---
title: AI Image Classifier
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.45.0
app_file: app.py
pinned: false
python_version: "3.11"
---

# 🧠 AI-Based Image Classifier

> Computer Vision project using Convolutional Neural Networks (CNN) trained on CIFAR-10 dataset.  
> Built as part of **AI/ML Internship — LMS Trainee Program**

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/ai-image-classifier)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-ff4b4b)](https://streamlit.io)

---

## 🎯 Project Overview

This project builds an AI model that can classify images into **10 categories** using a custom Convolutional Neural Network (CNN). The model is trained on the **CIFAR-10 dataset** — one of the most widely used benchmarks in computer vision.

| Detail | Value |
|--------|-------|
| **Developer** | Uttam Kumar |
| **Internship** | AI/ML Trainee Program — LMS (via LinkedIn) |
| **Domain** | Computer Vision / Deep Learning |
| **Dataset** | CIFAR-10 (60,000 images) |
| **Model** | Custom CNN (3 Conv Blocks) |
| **Framework** | PyTorch + Streamlit |
| **Test Accuracy** | ~82% |

---

## 🖼️ 10 Classes

| ✈️ Airplane | 🚗 Automobile | 🐦 Bird | 🐱 Cat | 🦌 Deer |
|:-----------:|:-------------:|:-------:|:------:|:-------:|
| 🐶 Dog | 🐸 Frog | 🐴 Horse | 🚢 Ship | 🚛 Truck |

---

## 🚀 Live Demo

👉 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/ai-image-classifier)**

Upload any image → AI predicts the class with confidence score!

---

## 📁 Project Structure

```
ai-image-classifier/
│
├── app.py                            # Streamlit web app
├── image_classifier_pytorch.py       # Model training script
├── requirements.txt                  # Python dependencies
├── runtime.txt                       # Python version (3.11)
├── best_model.pth                    # Trained model weights
└── README.md                         # This file
```

---

## ⚙️ How to Run Locally

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-image-classifier.git
cd ai-image-classifier
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Train the model (optional — skip if best_model.pth exists)
```bash
python image_classifier_pytorch.py
```
> ⏳ Training takes ~30-60 min on CPU, ~10 min on GPU.  
> After training, `best_model.pth` will be saved automatically.

### Step 4 — Run the web app
```bash
streamlit run app.py
```
> Open `http://localhost:8501` in your browser.

---

## 🏗️ Model Architecture

```
Input (32x32 RGB Image)
        ↓
Conv Block 1 — Conv2D(32) + BatchNorm + ReLU × 2 + MaxPool + Dropout(0.25)
        ↓
Conv Block 2 — Conv2D(64) + BatchNorm + ReLU × 2 + MaxPool + Dropout(0.25)
        ↓
Conv Block 3 — Conv2D(128) + BatchNorm + ReLU × 2 + MaxPool + Dropout(0.25)
        ↓
Flatten → Dense(256) + BatchNorm + ReLU + Dropout(0.5)
        ↓
Output — Dense(10) + Softmax
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~91% |
| Validation Accuracy | ~83% |
| **Test Accuracy** | **~82%** |
| Test Loss | ~0.54 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Airplane | 85% | 83% | 84% |
| Automobile | 90% | 91% | 90% |
| Bird | 75% | 74% | 74% |
| Cat | 68% | 65% | 66% |
| Deer | 83% | 85% | 84% |
| Dog | 73% | 72% | 72% |
| Frog | 88% | 89% | 88% |
| Horse | 87% | 88% | 87% |
| Ship | 90% | 92% | 91% |
| Truck | 88% | 87% | 87% |

---

## 📦 Dependencies

```
torch==2.10.0
torchvision==0.25.0
streamlit==1.45.0
numpy==2.1.0
matplotlib==3.9.0
Pillow==10.4.0
scikit-learn==1.5.0
```

---

## 📂 Dataset

- **Name:** CIFAR-10 (Canadian Institute for Advanced Research)
- **Size:** 60,000 color images (32×32 pixels)
- **Split:** 50,000 train | 10,000 test
- **Classes:** 10 (balanced — 6,000 images per class)
- **Kaggle:** https://www.kaggle.com/c/cifar-10
- **Official:** https://www.cs.toronto.edu/~kriz/cifar.html

> Dataset is downloaded automatically via `torchvision.datasets.CIFAR10(download=True)`

---

## 🔮 Future Improvements

- [ ] Transfer Learning using ResNet50 / MobileNetV2 for 90%+ accuracy
- [ ] Real-time camera prediction
- [ ] Grad-CAM visualization to see what the model focuses on
- [ ] Deploy on AWS / Google Cloud for production scale
- [ ] Extend to object detection using YOLO

---

## 👤 Author

**Uttam Kumar**  
AI/ML Internship Trainee — LMS Trainee Program (via LinkedIn)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
