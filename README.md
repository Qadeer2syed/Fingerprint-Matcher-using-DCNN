# Fingerprint Matcher using DCNN

**Fingerprint Matcher using DCNN** is a project that leverages Deep Convolutional Neural Networks (DCNN) to perform fingerprint pattern matching. This approach aims to enhance the accuracy and efficiency of fingerprint recognition systems by utilizing deep learning techniques.

---

## 🧠 Overview

This project implements a DCNN-based model tailored for fingerprint matching tasks. By learning intricate patterns and features from fingerprint images, the model can effectively distinguish between different fingerprints, facilitating reliable authentication and identification processes.

---

## 📁 Project Structure

```
Fingerprint-Matcher-using-DCNN/
├── data/                     # Directory containing fingerprint datasets
├── datasets.py               # Script for data loading and preprocessing
├── evaluate_matcher.py       # Script to evaluate the trained model
├── losses.py                 # Custom loss functions for training
├── model.py                  # DCNN model architecture
├── train_matcher.py          # Script to train the model
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.6 or higher
- PyTorch
- NumPy
- OpenCV
- Matplotlib

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Qadeer2syed/Fingerprint-Matcher-using-DCNN.git
cd Fingerprint-Matcher-using-DCNN
```

2. **Install dependencies:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧪 Usage

### Training

Train the model using the provided training script:

```bash
python train_matcher.py
```

### Evaluation

Evaluate the trained model on test images:

```bash
python evaluate_matcher.py
```

*The evaluation results will be displayed in the console or saved as specified in the script.*

---

## 📊 Results

The model demonstrates effective matching of fingerprint patterns, showcasing the potential of DCNNs in biometric authentication systems.

---

