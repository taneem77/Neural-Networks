Here is your document properly formatted as a clean **`.md`**** file (raw Markdown)** with all structure preserved and fixed:

````md
# Perceptron CLI — From-Scratch ML to Linux Kernel Inference

A single-layer perceptron classifier implemented from scratch with NumPy and developed progressively from basic logic-gate experiments into a complete command-line machine learning workflow and an initial **Linux kernel-space inference prototype**.

The project currently supports:

- Perceptron training from scratch
- Evaluation and manual metrics
- Prediction
- Model persistence
- Experiment tracking
- Hyperparameter tuning
- Interactive terminal training
- Structured CLI commands
- Linux kernel module compilation
- Fixed-point perceptron inference inside kernel space

The core perceptron algorithm and evaluation metrics are implemented manually rather than using pre-built ML classifiers or `sklearn.metrics`.

---

## Overview

The goal of this project is to understand what happens inside a perceptron instead of treating machine learning as a black box.

The project began with a minimal implementation capable of learning simple logic gates and was gradually extended with convergence handling, real-world data preprocessing, evaluation metrics, model persistence, experiment logging, and a structured CLI.

The current ML classifier operates on a binary subset of the Iris dataset:

```
setosa vs versicolor
```

The latest stage extends the project toward **ML + systems integration** by reproducing the lightweight perceptron inference operation inside a Linux loadable kernel module written in C.

The overall progression is:

```
Perceptron Fundamentals
        ↓
Training Improvements
        ↓
Iris Data Pipeline
        ↓
Interactive Training
        ↓
Evaluation & Persistence
        ↓
Structured CLI
        ↓
Linux Kernel Module
        ↓
Python ↔ Kernel Integration
```

---

# Project Progress

## Stage 1 — Perceptron Fundamentals

The project started with a basic perceptron implemented from scratch using Python and NumPy.

The initial model included:

- Weight initialization
- Bias handling
- Weighted-sum calculation
- Step activation
- Prediction
- Perceptron learning rule
- Epoch-based training

For an input vector:

```
x = [x1, x2, ..., xn]
```

the perceptron calculates:

```
z = w · x + b
```

and applies a binary step activation:

```
prediction = 1    if z >= 0
prediction = 0    otherwise
```

When a sample is misclassified:

```
error = actual - predicted

w = w + learning_rate × error × x
b = b + learning_rate × error
```

The initial model was tested on simple problems including **AND and OR gates**.

These experiments demonstrated an important limitation of a single-layer perceptron: it can only learn **linearly separable decision boundaries**.

---

## Stage 2 — Training Improvements

The initial implementation was extended to make training more stable, observable, and efficient.

Added:

- Error tracking across epochs
- Accuracy tracking
- Best-weight preservation
- Early stopping
- Learning-rate decay
- Confidence-scaled updates
- Convergence monitoring

Instead of blindly retaining the final epoch's parameters, the model can preserve its best-performing weights and stop training once convergence is detected.

### Learning Rate Decay

Instead of keeping the learning rate constant:

```
current_lr = initial_lr × decay^epoch
```

This allows larger updates during the beginning of training and progressively smaller updates as the model approaches convergence.

### Early Stopping

Training does not necessarily need to run for every configured epoch.

Example:

```
Maximum epochs: 100
Patience: 3

Perfect epoch
Perfect epoch
Perfect epoch
→ Stop training
```

This avoids unnecessary computation after convergence.

---

## Stage 3 — Iris Data Pipeline

The model was then moved from synthetic logic-gate examples to a real dataset.

The Iris dataset is filtered into the binary classification problem:

```
setosa vs versicolor
```

The data pipeline includes:

- 100 samples
- 4 numerical input features
- Binary class filtering
- 80/20 train-test split
- Feature standardization

The four features are:

```
sepal length (cm)
sepal width (cm)
petal length (cm)
petal width (cm)
```

Feature standardization is important because features with larger numerical ranges could otherwise dominate the perceptron's weight updates.

The transformation is:

```
x_scaled = (x - mean) / std
```

`scikit-learn` is used for dataset loading, train-test splitting, and preprocessing.

The **classifier itself is implemented from scratch**.

---

## Stage 4 — Interactive Training Interface

An interactive terminal interface was added to make the training process visible instead of hiding it behind a single function call.

Run:

```bash
python cli.py train
```

The interface displays:

- Dataset information
- Selected classes
- Train/test split
- Model configuration
- Learning rate
- Maximum epochs
- Learning-rate decay
- Early-stopping patience
- Training progress
- Train accuracy
- Test accuracy
- Learning curve
- Sample predictions

The model can also be retrained interactively using different hyperparameters.

---

## Stage 5 — Structured CLI

The project was reorganized into a git-style command-line interface with separate commands for the different parts of the ML workflow.

```bash
python cli.py train

python cli.py predict --features 5.1 3.5 1.4 0.2

python cli.py eval --verbose

python cli.py history

python cli.py info
```

This separates training, inference, evaluation, model inspection, and experiment tracking instead of handling everything inside a single script.

---

# Training

The model can be trained directly using:

```bash
python cli.py train
```

Hyperparameters can also be changed.

Example:

```bash
python cli.py train --lr 0.05
```

Training tracks:

```
errors
accuracy
learning rate
epochs
learned weights
bias
```

Early stopping prevents unnecessary epochs after convergence.

---

# Evaluation

Model evaluation is available through:

```bash
python cli.py eval --verbose
```

The evaluation metrics are calculated **manually from model predictions** rather than using `sklearn.metrics`.

Implemented metrics include:

- Confusion matrix
- Accuracy
- Precision
- Recall
- F1 score
- True Positives
- True Negatives
- False Positives
- False Negatives

### Confusion Matrix

```
TP — True Positive
TN — True Negative
FP — False Positive
FN — False Negative
```

### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

```
Precision = TP / (TP + FP)
```

### Recall

```
Recall = TP / (TP + FN)
```

### F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

# Prediction

New samples can be classified directly from the terminal.

Example:

```bash
python cli.py predict --features 6.3 2.9 4.7 1.6
```

The values represent:

```
sepal length
sepal width
petal length
petal width
```

Before inference, the sample is standardized using the same normalization statistics learned from the training data.

The CLI also displays a prediction margin representing how far the sample lies from the perceptron's decision boundary.

---

# Model Persistence

A trained model can be stored and reused instead of being retrained for every prediction.

The saved model includes:

- Learned weights
- Bias
- Feature means
- Feature standard deviations
- Model metadata

---

# Experiment History

Training experiments can be recorded and compared across sessions.

Example:

```bash
python cli.py history --log --lr 0.1
python cli.py history --log --lr 0.05
python cli.py history --log --lr 0.01
python cli.py history
```

Each run records:

- Timestamp
- Learning rate
- Epoch configuration
- Epochs completed
- Test accuracy
- Learned weights

---

# Stage 6 — Linux Kernel Integration

The latest stage moves beyond the Python ML workflow and explores how lightweight perceptron inference can execute at the **operating-system level**.

The key design decision is:

> **Training and preprocessing remain in Python/user space. Only lightweight inference is currently being explored in Linux kernel space.**

Kernel implementation:

```
src/kernel/
├── perceptron_kernel.c
├── perceptron_model.h
├── Makefile
└── README.md
```

---

## Kernel-Space Perceptron

The same operation:

```
z = w · x + b
```

is implemented in C inside a Linux kernel module.

Pipeline:

```
Input Features
      ↓
Weighted Sum
      ↓
w · x + bias
      ↓
Step Activation
      ↓
Binary Prediction
```

---

## Fixed-Point Model Representation

To avoid floating-point operations:

```
scale = 1000
```

Example:

```
0.5  →  500
1.1  → 1100
-0.2 → -200
```

---

# Building the Kernel Module

Tested on:

```
Ubuntu Linux
Kernel: 6.2.0-39-generic
x86_64
```

Build:

```bash
cd src/kernel
make
```

Output:

```
perceptron_kernel.ko
```

---

# Running Kernel Inference

Load module:

```bash
sudo insmod perceptron_kernel.ko
```

Check:

```bash
lsmod | grep perceptron
```

Logs:

```bash
sudo dmesg | grep perceptron
```

Unload:

```bash
sudo rmmod perceptron_kernel
```

---

# System Architecture

```
USER SPACE
──────────
Iris Dataset
    ↓
Train/Test Split
    ↓
Standardization
    ↓
Perceptron Training
    ↓
Weights + Bias
    ↓
Fixed-Point Conversion
    ↓
KERNEL SPACE
────────────
w · x + b
    ↓
Step Activation
    ↓
Binary Output
```

---

# CLI Commands

| Command | Purpose |
|--------|--------|
| train | Train model |
| predict | Run inference |
| eval | Evaluate model |
| history | View experiments |
| info | Model metadata |

---

# Project Status

```
Perceptron Fundamentals
        ↓
Optimized Training
        ↓
Iris Classification
        ↓
Evaluation System
        ↓
Experiment Tracking
        ↓
Structured CLI
        ↓
Linux Kernel Module
        ↓
Fixed-Point Inference
        ↓
Python ↔ Kernel Integration (IN PROGRESS)
```

---

# What's Next

- Export trained weights from Python → kernel
- Auto-quantization pipeline
- User-space ↔ kernel communication
- Match Python vs kernel predictions
- Performance benchmarking
- Explore eBPF inference
- Extend to multi-layer perceptron
- Implement backpropagation from scratch
````
