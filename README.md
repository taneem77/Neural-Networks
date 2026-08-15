# Neural Networks from Scratch

> A modular implementation of a **Single-Layer Perceptron** built during my Machine Learning internship, progressing from mathematical foundations to Linux kernel-space inference and multi-dataset benchmarking.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Linear%20Algebra-orange?style=for-the-badge&logo=numpy)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Benchmark-F7931E?style=for-the-badge&logo=scikitlearn)
![Linux](https://img.shields.io/badge/Linux-Kernel-green?style=for-the-badge&logo=linux)

---

## Project Overview

This project implements a **Perceptron classifier entirely from scratch** using NumPy and gradually extends it into a complete experimentation framework.

Rather than relying on machine learning libraries for training, the project focuses on understanding the learning algorithm mathematically before comparing it against Scikit-Learn's implementation.

### Features

- Custom Perceptron implemented from scratch
- Learning rate optimization & early stopping
- Interactive CLI dashboard
- Model persistence (`.npz`)
- Experiment logging
- Linux kernel-space inference prototype
- Multi-dataset benchmarking
- Confusion matrices & performance metrics
- Convergence visualization
- Decision boundary visualization

---

## Repository Structure

```text
Neural-Networks/
│
├── src/
│   ├── cli.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── persist.py
│   ├── logger.py
│   ├── terminal_ui.py
│   ├── perceptron_core.py
│   ├── optimised_perceptron.py
│   ├── datasets.py
│   ├── metrics.py
│   ├── visualizer.py
│   └── benchmark.py
│
├── kernel/
│   ├── kernel_bridge.py
│   ├── perceptron_kernel.c
│   ├── perceptron_kmod.c
│   ├── perceptron_kmod.h
│   └── perceptron_model.h
│
├── outputs/
│   ├── benchmark/
│   ├── confusion/
│   ├── convergence/
│   ├── boundaries/
│   └── report.csv
│
├── logs/
├── requirements.txt
└── README.md
```

---

## Learning Pipeline

```text
Dataset
   │
   ▼
Standardization
   │
   ▼
Perceptron Training
   │
   ▼
Weight Updates
   │
   ▼
Evaluation Metrics
   │
   ▼
Visualization & Benchmarking
```

---

<<<<<<< HEAD
# Tech Stack
=======
## Datasets Evaluated

| Dataset | Type | Linear Separability |
|----------|------|--------------------|
| AND | Logical Gate | Yes |
| OR | Logical Gate | Yes |
| NAND | Logical Gate | Yes |
| XOR | Logical Gate | No |
| Iris | Botanical | Yes |
| Breast Cancer | Medical | Mostly |
| Wine | Chemical | Mostly |
| Digits (0 vs 1) | Vision | Yes |

---

# Benchmark Results

The custom implementation was evaluated against Scikit-Learn's Perceptron.

| Dataset | Custom Accuracy | Scikit-Learn |
|----------|----------------|--------------|
| AND | 100% | — |
| OR | 100% | — |
| NAND | 100% | — |
| XOR | 50% | — |
| Iris | 100% | 100% |
| Breast Cancer | 97.4% | 95.6% |
| Wine | 88.5% | 96.2% |
| Digits | 100% | 98.6% |

---

## Accuracy Comparison

![Accuracy Comparison](outputs/benchmark/accuracy_comparison.png)

The benchmark demonstrates that the custom Perceptron achieves performance comparable to Scikit-Learn on linearly separable datasets while correctly exposing its limitations on more challenging data.

---

## Scikit-Learn Comparison

![Sklearn Comparison](outputs/benchmark/sklearn_vs_custom.png)

---

# Convergence Analysis

The model records the number of misclassifications after every epoch, allowing visualization of learning behaviour.

### Iris Dataset

![Iris Convergence](outputs/convergence/Iris.png)

### Breast Cancer Dataset

![Breast Cancer Convergence](outputs/convergence/Breast_Cancer.png)

---

# Confusion Matrices

### Iris

![Iris Confusion Matrix](outputs/confusion/Iris.png)

### Breast Cancer

![Breast Cancer Confusion Matrix](outputs/confusion/Breast_Cancer.png)

### Wine

![Wine Confusion Matrix](outputs/confusion/Wine.png)

---

# Decision Boundary Visualizations

The following datasets contain two-dimensional feature spaces, making them ideal for visualizing the learned separating hyperplane.

### AND Gate

![AND Boundary](outputs/boundaries/AND.png)

### OR Gate

![OR Boundary](outputs/boundaries/OR.png)

### NAND Gate

![NAND Boundary](outputs/boundaries/NAND.png)

### XOR

![XOR Boundary](outputs/boundaries/XOR.png)

The XOR visualization illustrates the fundamental limitation of a **single-layer perceptron**: the classes are **not linearly separable**, so no linear decision boundary can perfectly classify every sample.

---

# Linux Kernel Prototype

Beyond the Python implementation, the project includes a Linux kernel-space prototype demonstrating how learned perceptron weights can be used for inference through a character device and `ioctl` communication.

**Architecture**

```text
Python Training
       │
       ▼
 Learned Weights
       │
       ▼
 Fixed-Point Conversion
       │
       ▼
Kernel Character Device
       │
       ▼
Kernel Dot Product
       │
       ▼
Prediction
```

This prototype showcases communication between user space and kernel space while performing the weighted summation inside the Linux kernel.

---

# Running the Project
>>>>>>> 60f5210 (updated)

## Install dependencies

```bash
pip install -r requirements.txt
```

## Train the model

```bash
cd src
python cli.py train
```

## Evaluate

```bash
python cli.py eval
```

## Interactive prediction

```bash
python cli.py predict --interactive
```

## Run full benchmark

```bash
python cli.py benchmark
```

This automatically generates:

- Confusion matrices
- Convergence plots
- Decision boundaries
- Accuracy benchmark
- Scikit-Learn comparison
- `outputs/report.csv`

---

# Future Improvements

- Multi-layer Perceptron implementation
- Gradient descent with differentiable activations
- CUDA/GPU acceleration
- Additional real-world datasets
- Model export to embedded systems

---


**Tanisha Mathur**

Computer Science & Engineering • PES University

Machine Learning Internship Project