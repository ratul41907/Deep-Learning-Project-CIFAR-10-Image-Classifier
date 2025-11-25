# CIFAR-10 Image Classification with Dense Neural Network

A deep learning project that demonstrates image classification using a fully connected neural network on the CIFAR-10 dataset.

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Goal](#project-goal)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Features](#features)
- [Dependencies](#dependencies)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

---

## 📖 Introduction

This project builds a deep neural network using Keras and TensorFlow to classify images from the CIFAR-10 dataset. The network is fully connected (dense) and uses techniques such as batch normalization and dropout for better generalization.

---

## 📂 Dataset

**CIFAR-10** is a widely-used dataset in machine learning, consisting of 60,000 32×32 color images in 10 different classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The dataset is split into 50,000 training images and 10,000 test images.

---

## 🎯 Project Goal

To build and train a dense neural network capable of classifying CIFAR-10 images into their respective categories with high accuracy.

---

## 🛠️ Installation

1. Clone this repository or download the script [ostad_project.py](./ostad_project.py)

2. Install the required Python packages:

```bash
pip install tensorflow numpy matplotlib
```

---

## 🚀 Usage

You can run the entire pipeline by executing the Python script:

```bash
python ostad_project.py
```

This will:

- Load and visualize the CIFAR-10 dataset.
- Normalize and preprocess the data.
- Train a neural network.
- Plot training/validation accuracy and loss.
- Evaluate the model on test data.
- Show prediction results for random samples.

---

## 🧠 Model Architecture

The model is a Sequential dense neural network:

- **Input Layer**: Flattened 32x32x3 image vector (3072 features)
- **Dense Layer**: 512 neurons + ReLU + BatchNormalization + Dropout(0.3)
- **Dense Layer**: 256 neurons + ReLU + BatchNormalization + Dropout(0.3)
- **Dense Layer**: 128 neurons + ReLU + BatchNormalization + Dropout(0.3)
- **Output Layer**: 10 neurons (Softmax for classification)

**Loss**: Categorical Crossentropy  
**Optimizer**: Adam  
**Metrics**: Accuracy

---

## 📈 Training & Evaluation

- **Epochs**: 15  
- **Batch Size**: 64  
- **Validation Split**: 20%  
- **Final Test Accuracy**: ~4.6%

The project includes plots for:

- Training vs Validation Accuracy
- Training vs Validation Loss

---

## ✨ Features

- Fully connected Dense Neural Network
- Batch Normalization and Dropout for regularization
- Data normalization and preprocessing
- Real-time performance visualization
- Easy-to-understand architecture

---

## 📦 Dependencies

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## 🧪 Examples

Below is an example of prediction output:

```
True: cat     | Predicted: dog
True: truck   | Predicted: automobile
```

Visual output will show original images with both predicted and true labels.

---

## 🐞 Troubleshooting

- **Low Accuracy**: Dense-only networks don't capture spatial features well. Consider using CNNs.
- **Memory Issues**: Reduce batch size or number of layers.
- **Long Training Time**: Try fewer epochs or smaller hidden layers.

---

## 👥 Contributors

- Arafat Zaman Ratul

Feel free to contribute by submitting pull requests or reporting issues.

---

