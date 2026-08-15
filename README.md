# Neural Networks from Scratch

<div align="center">

A modular implementation of a **Single-Layer Perceptron** built during my Machine Learning Internship, progressing from mathematical foundations to Linux kernel-space inference and comprehensive benchmarking.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Linear%20Algebra-orange?style=for-the-badge&logo=numpy)
![Linux](https://img.shields.io/badge/Linux-Kernel-green?style=for-the-badge&logo=linux)
![Scikit-Learn](https://img.shields.io/badge/Benchmark-ScikitLearn-F7931E?style=for-the-badge&logo=scikitlearn)

</div>

---

## Overview

This project implements a **Single-Layer Perceptron entirely from scratch** using NumPy and gradually evolves it into a complete experimentation framework.

Rather than treating the perceptron as a black box, every component—from weight updates and prediction to convergence analysis and evaluation metrics—is implemented manually. The project concludes with a Linux kernel-space inference prototype and benchmarking against Scikit-Learn.

### Highlights

- Perceptron implemented from scratch
- Learning rate optimisation
- Early convergence detection
- Interactive CLI dashboard
- Model persistence (`.npz`)
- Experiment logging
- Multi-dataset benchmarking
- Confusion matrices & F1 metrics
- Decision boundary visualisation
- Linux kernel-space inference prototype

---

# Repository Structure

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
├── assets/
│   └── outputs/
│       ├── benchmark/
│       ├── boundaries/
│       ├── confusion/
│       ├── convergence/
│       └── report.csv
│
├── logs/
├── requirements.txt
└── README.md
```

---

# Learning Pipeline

```text
Dataset
   │
   ▼
Feature Standardization
   │
   ▼
Perceptron Training
   │
   ▼
Weight Optimisation
   │
   ▼
Evaluation Metrics
   │
   ▼
Visual Analytics
```

---

# Datasets

| Dataset | Domain | Linear? |
|----------|--------|----------|
| AND | Logic Gate | Yes |
| OR | Logic Gate | Yes |
| NAND | Logic Gate | Yes |
| XOR | Logic Gate | No |
| Iris | Botanical | Yes |
| Breast Cancer | Medical | Mostly |
| Wine | Chemical | Mostly |
| Digits (0 vs 1) | Vision | Yes |

---

# Benchmark Results

| Dataset | Custom | Scikit-Learn |
|----------|-------:|-------------:|
| AND | 100% | — |
| OR | 100% | — |
| NAND | 100% | — |
| XOR | 50% | — |
| Iris | 100% | 100% |
| Breast Cancer | 97% | ~98% |
| Wine | 94% | ~97% |
| Digits | 100% | 100% |

---

# Accuracy Benchmark

Overall classification accuracy across every evaluated dataset.

![Accuracy](assets/outputs/benchmark/accuracy_comparison.png)

---

# Scikit-Learn Comparison

Comparison between the custom implementation and `sklearn.linear_model.Perceptron`.

![Sklearn](assets/outputs/benchmark/sklearn_vs_custom.png)

---

# Convergence Analysis

The model records the number of misclassifications after every epoch, allowing convergence behaviour to be visualised.

| Iris | Breast Cancer |
|------|------|
| ![](assets/outputs/convergence/Iris.png) | ![](assets/outputs/convergence/Breast_Cancer.png) |

| Wine | Digits |
|------|------|
| ![](assets/outputs/convergence/Wine.png) | ![](assets/outputs/convergence/Digits.png) |

---

# Confusion Matrices

These matrices illustrate exactly where the classifier succeeds and where it makes mistakes.

| Iris | Breast Cancer |
|------|------|
| ![](assets/outputs/confusion/Iris.png) | ![](assets/outputs/confusion/Breast_Cancer.png) |

| Wine | Digits |
|------|------|
| ![](assets/outputs/confusion/Wine.png) | ![](assets/outputs/confusion/Digits.png) |

---

# Decision Boundaries

The learned separating hyperplane for two-dimensional datasets.

| AND | OR |
|-----|----|
| ![](assets/outputs/boundaries/AND.png) | ![](assets/outputs/boundaries/OR.png) |

| NAND | XOR |
|------|------|
| ![](assets/outputs/boundaries/NAND.png) | ![](assets/outputs/boundaries/XOR.png) |

---

# Why XOR Fails

A **Single-Layer Perceptron** can only learn **linearly separable** datasets.

The XOR dataset is not linearly separable, meaning no straight decision boundary can perfectly classify every sample. This limitation directly motivates **Multilayer Perceptrons (MLPs)** and deeper neural networks.

---

# Linux Kernel Inference Prototype

Alongside the Python implementation, this project includes a **Linux kernel-space prototype** demonstrating how learned perceptron weights can be used for inference through a character device and `ioctl` communication.

## Architecture

```text
                USER SPACE

      Python Perceptron (NumPy)
               │
               ▼
      Learned Weights & Bias
               │
               ▼
      Fixed-Point Conversion
               │
               ▼
        kernel_bridge.py
               │
          ioctl() System Call
               │
────────────────────────────────────────

               KERNEL SPACE

     Character Device Driver
      (/dev/perceptron_kmod)
               │
               ▼
     Fixed-Point Dot Product
               │
               ▼
      Activation  z = wx + b
               │
               ▼
        Binary Classification
```

## Kernel Components

| File | Purpose |
|------|---------|
| `kernel_bridge.py` | Python ↔ Kernel communication layer |
| `perceptron_kmod.c` | Character device implementing `ioctl` |
| `perceptron_kernel.c` | Kernel-space perceptron inference |
| `perceptron_model.h` | Fixed-point weights & bias |
| `perceptron_kmod.h` | Shared ioctl interface |

The kernel module computes the weighted summation using **Q16.16 fixed-point arithmetic**, providing a lightweight proof-of-concept inference path outside traditional user-space execution.

---

# Running the Project

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
cd src
python cli.py train
```

## Evaluate

```bash
python cli.py eval
```

## Interactive Prediction

```bash
python cli.py predict --interactive
```

## Run Full Benchmark

```bash
python cli.py benchmark
```

The benchmark automatically generates:

- Accuracy comparison
- Scikit-Learn comparison
- Convergence plots
- Confusion matrices
- Decision boundaries
- `assets/outputs/report.csv`

---

# Technologies

| Category | Stack |
|----------|------|
| Language | Python, C |
| ML | NumPy, Scikit-Learn |
| Visualisation | Matplotlib |
| CLI | argparse |
| Persistence | NumPy `.npz` |
| Logging | Python Logging |
| Kernel | Linux Character Device + ioctl |

---

# Future Improvements

- Multilayer Perceptron
- Backpropagation
- ReLU & Sigmoid activations
- GPU acceleration
- Embedded deployment
- Additional real-world datasets

---

# Author

**Tanisha Mathur**

Computer Science & Engineering • PES University

Machine Learning Internship Project