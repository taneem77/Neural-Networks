# Neural Networks from Scratch

<div align="center">

A modular implementation of a **Single-Layer Perceptron** built during my Machine Learning Internship, progressing from mathematical foundations to Linux kernel-space inference and comprehensive benchmarking.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Linear%20Algebra-orange?style=for-the-badge&logo=numpy)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Benchmark-F7931E?style=for-the-badge&logo=scikitlearn)
![Linux](https://img.shields.io/badge/Linux-Kernel-green?style=for-the-badge&logo=linux)

</div>

---

## Project Overview

This project implements a **Perceptron classifier entirely from scratch** using **NumPy**, gradually evolving into a complete experimentation framework with benchmarking, visualization, experiment tracking and a Linux kernel-space inference prototype.

Instead of relying on machine learning libraries for training, every weight update, prediction, convergence check and evaluation metric is implemented manually to understand the mathematics behind linear classifiers.

### Features

- Single-Layer Perceptron implemented from scratch
- Learning rate optimization
- Early stopping & convergence tracking
- Interactive CLI dashboard
- Model persistence (`.npz`)
- Experiment logging
- Linux kernel-space inference prototype
- Multi-dataset benchmarking
- Confusion matrices & performance metrics
- Decision boundary visualization
- Scikit-Learn comparison

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
│   ├── benchmark.py
│   └── outputs/
│
├── kernel/
│   ├── kernel_bridge.py
│   ├── perceptron_kernel.c
│   ├── perceptron_kmod.c
│   ├── perceptron_kmod.h
│   └── perceptron_model.h
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
Visual Analytics
```

---

# Multi-Dataset Benchmark

The project evaluates the custom perceptron on both logical gate datasets and real-world classification datasets.

| Dataset | Type | Linear? |
|----------|------|----------|
| AND | Logic Gate | Yes |
| OR | Logic Gate | Yes |
| NAND | Logic Gate | Yes |
| XOR | Logic Gate | No |
| Iris | Botanical | Yes |
| Breast Cancer | Medical | Mostly |
| Wine | Chemical | Mostly |
| Digits (0 vs 1) | Vision | Yes |

---

# Experimental Results

| Dataset | Custom Perceptron | Scikit-Learn |
|----------|------------------|--------------|
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

The benchmark below compares the performance of the custom implementation across all datasets.

![Accuracy Benchmark](src/outputs/benchmark/accuracy_comparison.png)

---

# Scikit-Learn Comparison

Performance comparison between the custom implementation and Scikit-Learn's Perceptron.

![Scikit Comparison](src/outputs/benchmark/sklearn_vs_custom.png)

---

# Convergence Analysis

The number of misclassifications is recorded after every epoch to visualize how quickly the model converges.

## Iris Dataset

![Iris Convergence](src/outputs/convergence/Iris.png)

## Breast Cancer Dataset

![Breast Cancer Convergence](src/outputs/convergence/Breast_Cancer.png)

## Wine Dataset

![Wine Convergence](src/outputs/convergence/Wine.png)

---

# Confusion Matrices

These matrices illustrate exactly where the classifier succeeds and where it makes mistakes.

### Iris

![Iris CM](src/outputs/confusion/Iris.png)

### Breast Cancer

![Breast Cancer CM](src/outputs/confusion/Breast_Cancer.png)

### Wine

![Wine CM](src/outputs/confusion/Wine.png)

### Digits

![Digits CM](src/outputs/confusion/Digits.png)

---

# Decision Boundary Visualizations

For two-dimensional datasets, the learned separating hyperplane can be visualized directly.

### AND Gate

![AND Boundary](src/outputs/boundaries/AND.png)

### OR Gate

![OR Boundary](src/outputs/boundaries/OR.png)

### NAND Gate

![NAND Boundary](src/outputs/boundaries/NAND.png)

### XOR

![XOR Boundary](src/outputs/boundaries/XOR.png)

---

# Why XOR Fails

A **single-layer perceptron** can only learn **linearly separable** datasets.

The XOR dataset is not linearly separable, meaning no straight decision boundary can perfectly classify every sample. This limitation motivates the development of **Multilayer Perceptrons (MLPs)** and deeper neural networks.

---

# Linux Kernel Inference Prototype

Beyond the Python implementation, this project includes a Linux kernel-space prototype demonstrating how learned perceptron weights can be used for inference through a character device and `ioctl` communication.

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
            ioctl() Call
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
         Activation z = wx + b
                 │
                 ▼
             Binary Prediction
```

## Kernel Components

| File | Purpose |
|------|---------|
| `kernel_bridge.py` | Communication layer between Python and the Linux kernel |
| `perceptron_kmod.c` | Character device driver implementing `ioctl` |
| `perceptron_kernel.c` | Kernel-space perceptron inference |
| `perceptron_kmod.h` | Shared user/kernel communication interface |
| `perceptron_model.h` | Fixed-point weights and bias representation |

The kernel module computes the weighted summation using **Q16.16 fixed-point arithmetic**, demonstrating a lightweight inference path outside traditional user-space execution.

---

# Command Line Interface

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

## Experiment History

```bash
python cli.py history
```

## Multi-Dataset Benchmark

```bash
python cli.py benchmark
```

Running the benchmark automatically generates:

- Accuracy benchmark plots
- Scikit-Learn comparison
- Convergence curves
- Confusion matrices
- Decision boundaries
- `report.csv`

---

# Technologies Used

| Category | Tools |
|----------|------|
| Language | Python, C |
| ML | NumPy, Scikit-Learn |
| Visualization | Matplotlib |
| CLI | argparse |
| Persistence | NumPy `.npz` |
| Kernel | Linux Character Device, ioctl |
| Logging | Python logging |

---

# Future Improvements

- Multilayer Perceptron implementation
- Sigmoid & ReLU activation functions
- Gradient Descent & Backpropagation
- GPU acceleration
- Additional medical and financial datasets
- Embedded deployment using exported kernel parameters

---

# Author

**Tanisha Mathur**

Computer Science & Engineering • PES University

Machine Learning Internship Project