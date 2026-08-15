# Neural Networks from Scratch

> A modular implementation of a **Single-Layer Perceptron** built during my Machine Learning Internship, progressing from mathematical foundations to Linux kernel-space inference and multi-dataset benchmarking.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Linear%20Algebra-orange?style=for-the-badge&logo=numpy)
![Linux](https://img.shields.io/badge/Linux-Kernel-green?style=for-the-badge&logo=linux)
![Scikit](https://img.shields.io/badge/Scikit--Learn-Benchmark-F7931E?style=for-the-badge&logo=scikitlearn)

---

# Project Overview

This project implements a **Perceptron Neural Network completely from scratch** using only **NumPy**, before extending it into a complete experimentation framework.

The objective was not simply achieving classification accuracy, but understanding every mathematical component involved in linear neural networks—including weight updates, convergence, evaluation metrics, persistence, visualization, and even Linux kernel-space inference.

---

# Features

- Single-layer Perceptron from scratch
- Manual gradient-free weight updates
- Learning rate optimization
- Early convergence detection
- Interactive CLI dashboard
- Model persistence (.npz)
- Experiment logging
- Manual confusion matrix implementation
- Precision, Recall & F1 Score
- Multi-dataset benchmarking
- Decision boundary visualization
- Linux Kernel character-device inference

---

# Repository Structure

```text
Neural-Networks/
│
├── assets/
│
├── kernel/
│
├── src/
│   ├── benchmark.py
│   ├── cli.py
│   ├── datasets.py
│   ├── evaluate.py
│   ├── logger.py
│   ├── metrics.py
│   ├── optimised_perceptron.py
│   ├── perceptron_core.py
│   ├── persist.py
│   ├── predict.py
│   ├── terminal_ui.py
│   ├── train.py
│   ├── visualizer.py
│   └── outputs/
│
└── README.md
```

---

# Development Journey

| Stage | Objective |
|--------|-----------|
| Day 1 | Build the Perceptron mathematically |
| Day 2 | Optimize learning & convergence |
| Day 3 | Iris dataset + preprocessing |
| Day 4 | CLI, persistence & evaluation |
| Day 5 | Linux Kernel inference prototype |
| Extension | Multi-dataset benchmarking |

---

# CLI Interface

The project is controlled entirely through a modular command-line interface.

```bash
python cli.py --help
```

![CLI Help](assets/cli_help.png)

Available commands:

- `train`
- `predict`
- `eval`
- `history`
- `info`
- `benchmark`

---

# Training Dashboard

Training displays dataset information, preprocessing details, learning rate configuration, convergence, accuracy and predictions in real time.

```bash
python cli.py train
```

![Training](assets/train.png)

Features shown:

- Dataset summary
- Standardization
- Hyperparameters
- Progress bar
- Learning curve
- Sample predictions

---

# Prediction

The trained model accepts custom flower measurements and predicts the species together with the decision boundary margin.

```bash
python cli.py predict --features 6.3 2.9 4.7 1.6
```

![Prediction](assets/predict.png)

Output includes:

- Predicted class
- Raw activation (z-score)
- Confidence margin
- Position relative to the decision boundary

---

# Evaluation Metrics

Evaluation is implemented manually without relying on Scikit-Learn metrics.

```bash
python cli.py eval --verbose
```

![Evaluation](assets/eval.png)

The project computes:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- TP / TN / FP / FN

along with detailed explanations of what each metric represents.

---

# Experiment History

Every training session can be permanently logged.

```bash
python cli.py history
```

![History](assets/history.png)

Logged information includes:

- Timestamp
- Epochs
- Learning rate
- Test accuracy
- Learned weights

The interactive dashboard also allows retraining directly from history.

![History Dashboard](assets/history_log.png)

---

# Benchmarking

The project was extended beyond Iris and evaluated on **8 datasets**.

| Dataset | Custom | Scikit-Learn |
|----------|--------|--------------|
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

Automatically generated after:

```bash
python cli.py benchmark
```

![Benchmark](src/outputs/benchmark/accuracy_comparison.png)

---

## Scikit-Learn Comparison

![Sklearn](src/outputs/benchmark/sklearn_vs_custom.png)

---

# Convergence Analysis

### Iris

![Iris](src/outputs/convergence/Iris.png)

### Breast Cancer

![Breast Cancer](src/outputs/convergence/Breast_Cancer.png)

### Wine

![Wine](src/outputs/convergence/Wine.png)

---

# Confusion Matrices

### Iris

![Iris CM](src/outputs/confusion/Iris.png)

### Breast Cancer

![Breast Cancer CM](src/outputs/confusion/Breast_Cancer.png)

### Wine

![Wine CM](src/outputs/confusion/Wine.png)

### Digits

![Digits CM](src/outputs/confusion/Digits.png)

---

# Decision Boundary Visualization

Logical gate datasets are visualized to show how a linear classifier separates classes.

### AND

![AND](src/outputs/boundaries/AND.png)

### OR

![OR](src/outputs/boundaries/OR.png)

### NAND

![NAND](src/outputs/boundaries/NAND.png)

### XOR

![XOR](src/outputs/boundaries/XOR.png)

The XOR dataset demonstrates the fundamental limitation of a **single-layer perceptron**: it is **not linearly separable**, therefore no linear decision boundary can perfectly classify it.

---

# Linux Kernel Inference Prototype

One of the extensions of this internship project is moving inference into **kernel space** through a Linux character device.

Instead of computing the dot product entirely in Python, feature vectors are transferred into the kernel using `ioctl`, where the perceptron performs fixed-point inference.

## Kernel Architecture

```text
User Space (Python)

Feature Vector
      │
      ▼
kernel_bridge.py
      │
      ▼
 ioctl()
      │
      ▼

────────────────────────────

 Linux Kernel

 Character Device
      │
      ▼
 Fixed-Point Dot Product
      │
      ▼
 Activation
      │
      ▼
 Binary Prediction
```

---

## Building the Kernel Module

```bash
cd kernel
make
```

![Kernel Build](assets/kernel_build.png)

---

## Loading the Module

```bash
sudo insmod perceptron_kernel.ko
```

The character device becomes available as:

```text
/dev/perceptron_kmod
```

![Module Loaded](assets/kernel_module_loaded.png)

---

## Kernel-space Inference

The kernel performs the weighted summation and logs activation values through `dmesg`.

![Kernel Inference](assets/kernel_inference.png)

Example kernel logs:

- Fixed-point scaling
- Input vector
- Activation value
- Final prediction

---

## Python ↔ Kernel Bridge

The bridge verifies that the kernel implementation produces the same result as the NumPy implementation.

```bash
python kernel_bridge.py
```

![Kernel Bridge](assets/kernel_bridge_output.png)

Example:

| Implementation | Dot Product |
|---------------|------------:|
| NumPy | -0.800 |
| Kernel | -0.7999 |

The small difference is due to fixed-point arithmetic inside kernel space.

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

## Predict

```bash
python cli.py predict --interactive
```

## Evaluate

```bash
python cli.py eval --verbose
```

## Benchmark

```bash
python cli.py benchmark
```

Generates automatically:

- Accuracy comparison
- Scikit-Learn comparison
- Convergence plots
- Confusion matrices
- Decision boundaries
- CSV performance report

---

# Future Improvements

- Multi-Layer Perceptron
- Backpropagation
- ReLU & Sigmoid activations
- GPU acceleration
- MNIST handwritten digit classification
- Embedded deployment using exported kernel weights

---

# Author

**Tanisha Mathur**

Computer Science & Engineering • PES University

Machine Learning Internship Project