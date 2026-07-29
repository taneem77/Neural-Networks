# Perceptron CLI — From-Scratch ML to Linux Kernel Inference

A single-layer perceptron classifier implemented from scratch with NumPy and progressively developed from basic logic-gate experiments into a complete command-line machine-learning workflow.

The project now extends beyond the Python implementation into **Linux systems programming**, with a basic loadable kernel module that performs fixed-point perceptron inference inside kernel space.

The project currently supports:

- Perceptron training from scratch
- Evaluation and manually implemented metrics
- Prediction on unseen samples
- Model persistence
- Experiment tracking
- Hyperparameter tuning
- Interactive terminal training
- Structured CLI commands
- Linux kernel module compilation and loading
- Fixed-point perceptron inference in kernel space

The core classifier and evaluation metrics are implemented manually rather than using pre-built ML classifiers or `sklearn.metrics`.

---

## Overview

The goal of this project is to understand what happens inside a perceptron instead of treating machine learning as a black box.

The project began with a minimal implementation capable of learning simple logic gates and was gradually extended with convergence handling, real-world data preprocessing, evaluation metrics, model persistence, experiment logging, and a structured CLI.

The current classifier operates on a binary subset of the Iris dataset:

```text
setosa vs versicolor
```

The latest stage explores how the same lightweight inference operation can be represented at a lower systems level using a **Linux loadable kernel module written in C**.

### Development Progression

```text
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
Fixed-Point Kernel Inference
        ↓
Python ↔ Kernel Integration
        [IN PROGRESS]
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

```text
x = [x1, x2, ..., xn]
```

the perceptron calculates:

```text
z = w · x + b
```

and applies a binary step activation:

```text
prediction = 1    if z >= 0
prediction = 0    otherwise
```

When a sample is misclassified:

```text
error = actual - predicted

w = w + learning_rate × error × x
b = b + learning_rate × error
```

The initial model was tested on simple problems including **AND and OR gates**.

These experiments also demonstrate a fundamental limitation of a single-layer perceptron: it can only learn **linearly separable decision boundaries**.

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

Instead of blindly retaining parameters from the final epoch, the model can preserve its best-performing weights and stop once convergence is detected.

### Learning Rate Decay

The convergent perceptron can gradually reduce its learning rate:

```text
current_lr = initial_lr × decay^epoch
```

This allows larger updates early in training while reducing the update size as the model approaches convergence.

### Early Stopping

Training does not necessarily need to run for every configured epoch.

For example:

```text
Maximum epochs: 100
Patience: 3

Perfect epoch
Perfect epoch
Perfect epoch
→ Stop training
```

This prevents unnecessary computation after convergence.

---

## Stage 3 — Iris Data Pipeline

The model was then moved from synthetic logic-gate examples to a real dataset.

The Iris dataset is filtered into a binary classification problem:

```text
setosa vs versicolor
```

The data pipeline includes:

- 100 total samples
- 4 numerical input features
- Binary class filtering
- 80/20 train-test split
- Feature standardization

The four input features are:

```text
sepal length (cm)
sepal width (cm)
petal length (cm)
petal width (cm)
```

Feature standardization is important because features with larger numerical ranges could otherwise dominate the perceptron's weight updates.

The transformation is:

```text
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

After training, the model can also be retrained interactively with different hyperparameters.

### Training Output

![Training Interface](assets/train.png)

---

## Stage 5 — Structured CLI

The project was reorganized into a git-style command-line interface with separate subcommands for different parts of the ML workflow.

```bash
python cli.py train
python cli.py predict --features 5.1 3.5 1.4 0.2
python cli.py eval --verbose
python cli.py history
python cli.py info
```

This separates training, inference, evaluation, model inspection, and experiment tracking instead of handling everything inside one script.

### CLI Interface

![CLI Help](assets/cli_help.png)

---

# Training

The model can be trained directly from the command line:

```bash
python cli.py train
```

Training can also be configured using different hyperparameters.

For example:

```bash
python cli.py train --lr 0.05
```

The training process tracks:

```text
errors
accuracy
learning rate
epochs completed
learned weights
bias
```

Early stopping prevents unnecessary epochs once the classifier has converged.

---

# Evaluation

Model evaluation is available through:

```bash
python cli.py eval --verbose
```

The evaluation module calculates metrics manually from model predictions rather than using `sklearn.metrics`.

Implemented metrics include:

- Confusion matrix
- Accuracy
- Precision
- Recall
- F1 score
- True positives
- True negatives
- False positives
- False negatives

Verbose mode also explains what each metric represents instead of only printing the numerical value.

### Evaluation Output

![Evaluation Metrics](assets/eval.png)

### Confusion Matrix

For binary classification:

```text
TP — True Positive
TN — True Negative
FP — False Positive
FN — False Negative
```

These values form the basis of the remaining metrics.

### Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall

```text
Recall = TP / (TP + FN)
```

### F1 Score

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Implementing these manually makes the relationship between individual predictions and the final evaluation metrics explicit.

---

# Prediction

New samples can be classified directly from the terminal.

Example:

```bash
python cli.py predict --features 6.3 2.9 4.7 1.6
```

The four values correspond to:

```text
sepal length
sepal width
petal length
petal width
```

Before inference, the sample is standardized using the same normalization statistics learned from the training data.

The CLI also displays a prediction margin, indicating how far the sample lies from the perceptron's decision boundary.

### Prediction Output

![Prediction](assets/predict.png)

---

# Model Persistence

A trained model can be stored and reused instead of being retrained for every prediction.

The saved model includes:

- Learned weights
- Bias
- Feature means
- Feature standard deviations
- Model metadata

Saving preprocessing statistics alongside the model is important.

During training:

```text
x_scaled = (x - mean) / std
```

If a new sample were normalized using different statistics, it would exist in a different feature space from the one used during training.

For this reason, the original training mean and standard deviation are saved and reused during inference.

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

Each run records information such as:

- Timestamp
- Learning rate
- Epoch configuration
- Epochs actually completed
- Test accuracy
- Learned weights

This makes it possible to compare how different hyperparameter choices affect training behavior.

### Experiment History Output

![Training History](assets/history.png)

---

# Stage 6 — Linux Kernel Integration

The latest stage moves beyond the Python ML workflow and explores how lightweight perceptron inference can execute at the **operating-system level**.

The design separates the two responsibilities:

```text
USER SPACE
Training + preprocessing

KERNEL SPACE
Lightweight inference
```

Training is intentionally kept in Python rather than attempting to train the neural network inside the kernel.

The kernel implementation is located under:

```text
src/kernel/
├── perceptron_kernel.c
├── perceptron_model.h
├── Makefile
└── README.md
```

---

## Kernel-Space Perceptron

The same fundamental inference operation used by the Python model:

```text
z = w · x + b
```

has been reproduced in C inside a Linux loadable kernel module.

The kernel inference path is:

```text
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

### `perceptron_kernel.c`

Contains:

- Kernel module initialization
- Kernel module cleanup
- Weighted-sum calculation
- Step activation
- Demonstration inference
- Kernel logging using `pr_info`

### `perceptron_model.h`

Contains the current:

- Number of features
- Fixed-point scaling factor
- Prototype weights
- Prototype bias

### `Makefile`

Uses the Linux kernel build system to compile the implementation as an external loadable kernel module.

---

## Why Fixed-Point Arithmetic?

The Python implementation uses floating-point NumPy operations.

For the initial kernel prototype, model parameters and features are represented as **scaled integers**.

The current scale factor is:

```text
SCALE_FACTOR = 1000
```

For example:

```text
 0.5  →   500
 1.1  →  1100
-0.2  →  -200
```

This allows the initial kernel inference implementation to use integer arithmetic.

The weights and bias currently stored in `perceptron_model.h` are **prototype parameters** used to establish and test the kernel execution path.

They are not yet the automatically exported parameters from the trained Python model.

---

# Building the Kernel Module

The kernel prototype was built and tested on:

```text
Ubuntu Linux
Kernel: 6.2.0-39-generic
Architecture: x86_64
```

From the kernel directory:

```bash
cd src/kernel
make
```

The Linux kernel build process compiles the module and generates:

```text
perceptron_kernel.ko
```

### Successful Kernel Build

![Kernel Module Build](assets/kernel_build.png)

The generated `.ko` file is the loadable kernel object containing the perceptron inference implementation.

---

# Running Perceptron Inference in Kernel Space

The compiled module is loaded using:

```bash
sudo insmod perceptron_kernel.ko
```

Its presence in the running kernel can be verified using:

```bash
lsmod | grep perceptron
```

The inference output is written to the Linux kernel log:

```bash
sudo dmesg | grep perceptron
```

### Kernel Inference Output

![Kernel Perceptron Inference](assets/kernel_inference.png)

The current demonstration produced:

```text
Input       = [500, -200, 1100, 800]
Activation  = -2350000
Prediction  = 0
```

The binary step activation therefore behaves as expected:

```text
activation < 0
      ↓
prediction = 0
```

This confirms that the prototype successfully:

```text
Compiled as a Linux kernel module
        ↓
Loaded into the running kernel
        ↓
Executed the perceptron weighted sum
        ↓
Applied the step activation
        ↓
Produced a binary prediction
```

After execution, the module can be removed using:

```bash
sudo rmmod perceptron_kernel
```

---

# Current System Architecture

The project currently contains a complete Python ML pipeline and a working kernel-side inference prototype.

```text
                         USER SPACE

                       Iris Dataset
                            │
                            ▼
                  Binary Class Selection
                  setosa vs versicolor
                            │
                            ▼
                    Train / Test Split
                            │
                            ▼
                     Standardization
                            │
                            ▼
                  Convergent Perceptron
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
         Evaluation     Prediction      Persistence
             │              │              │
             ▼              ▼              ▼
        Confusion       New Flower      Saved Model
          Matrix         Samples       + Scaler Stats
             │
             ▼
      Precision / Recall
          / F1 / Accuracy
                            │
                            ▼
                     Weights + Bias
                            │
                     [Next Stage]
                            │
                            ▼
                  Fixed-Point Export

════════════════ USER / KERNEL BOUNDARY ════════════════

                            │
                            ▼
                       KERNEL SPACE
                            │
                            ▼
                 Perceptron Inference
                            │
                       w · x + b
                            │
                            ▼
                     Step Activation
                            │
                            ▼
                    Binary Prediction
```

The kernel inference path is working.

The connection between the trained Python model and the kernel representation is the next integration stage.

---

# CLI Commands

| Command | Purpose |
|---|---|
| `train` | Train the perceptron and open the interactive terminal UI |
| `predict` | Classify a flower from four feature measurements |
| `eval` | Display the confusion matrix and evaluation metrics |
| `history` | View or record previous training runs |
| `info` | Display metadata for the saved model |

To view the complete CLI reference:

```bash
python cli.py --help
```

![CLI Commands](assets/cli_help.png)

---

# Project Structure

```text
Neural-Networks/
│
├── assets/
│   ├── train.png
│   ├── cli_help.png
│   ├── eval.png
│   ├── predict.png
│   ├── history.png
│   ├── kernel_build.png
│   └── kernel_inference.png
│
├── src/
│   ├── Python perceptron and CLI modules
│   │
│   └── kernel/
│       ├── perceptron_kernel.c
│       ├── perceptron_model.h
│       ├── Makefile
│       └── README.md
│
├── .gitignore
└── README.md
```

---

# Implementation Highlights

### From-Scratch Classifier

The perceptron learning algorithm is implemented manually using NumPy rather than a pre-built classifier.

### Manual Evaluation Metrics

Accuracy, precision, recall, F1, and the confusion matrix are derived directly from predictions.

### Consistent Preprocessing

Training normalization statistics are stored and reused during inference, preventing preprocessing differences between training and prediction.

### Convergence Tracking

Errors, accuracy, and learning rate can be tracked across epochs.

### Model Persistence

Learned parameters and preprocessing statistics can be stored and reused for future predictions.

### Experiment Logging

Multiple runs can be recorded and compared instead of losing previous hyperparameter experiments.

### Command-Line Workflow

Training, evaluation, prediction, history, and model information are exposed through separate CLI subcommands.

### Linux Kernel Prototype

The perceptron's inference equation has now also been implemented in C as a Linux loadable kernel module using fixed-point integer arithmetic.

The module has been successfully compiled, loaded, executed, and unloaded on Ubuntu Linux.

---

# Current Limitations

The current ML model remains a **single-layer perceptron**.

Its decision boundary is therefore linear.

This makes it suitable for linearly separable classification problems such as the selected Iris classes, but it cannot learn non-linearly separable relationships.

A classic example is XOR:

```text
0 XOR 0 → 0
0 XOR 1 → 1
1 XOR 0 → 1
1 XOR 1 → 0
```

No single linear decision boundary can separate the two XOR classes.

The kernel implementation is also currently an **initial prototype**.

At this stage:

- Training remains in Python
- Preprocessing remains in user space
- Kernel parameters are prototype values
- The demonstration input is currently defined inside the module
- The Python CLI does not yet communicate directly with the kernel
- Trained Python parameters are not yet exported automatically

These limitations define the next development stage.

---

# What's Next

The immediate focus is completing the connection between the existing ML pipeline and the Linux kernel prototype.

### Python ↔ Kernel Integration

1. Export the trained perceptron weights and bias from Python.
2. Quantize the learned parameters into fixed-point values.
3. Automatically generate the kernel model representation.
4. Create a user-space ↔ kernel-space communication interface.
5. Pass standardized Iris samples from Python to the kernel.
6. Compare Python and kernel predictions for consistency.
7. Benchmark inference latency and communication overhead.
8. Explore eBPF as an alternative kernel execution mechanism.

### Neural Network Extension

After the systems-integration stage:

- Build a multi-layer perceptron from scratch
- Implement forward propagation
- Implement backpropagation manually
- Introduce non-linear activation functions
- Train on non-linearly separable problems such as XOR
- Add automated model and evaluation tests

---

# Tech Stack

| Area | Technologies |
|---|---|
| Machine Learning | Python, NumPy |
| Dataset & Preprocessing | scikit-learn |
| CLI | argparse |
| Systems Programming | C |
| Operating System | Linux / Ubuntu |
| Kernel Development | Linux Kernel Modules, Kbuild |
| Version Control | Git, GitHub |

---

# Example Workflow

## Python ML Workflow

```bash
# Train the model
python cli.py train

# Inspect the saved model
python cli.py info

# Make a prediction
python cli.py predict --features 6.3 2.9 4.7 1.6

# Evaluate the model
python cli.py eval --verbose

# View experiment history
python cli.py history
```

## Linux Kernel Workflow

```bash
cd src/kernel

# Build the kernel module
make

# Load it
sudo insmod perceptron_kernel.ko

# Verify that it is loaded
lsmod | grep perceptron

# View kernel-space inference
sudo dmesg | grep perceptron

# Unload the module
sudo rmmod perceptron_kernel
```

---

# Status

**Current stage:** Single-layer perceptron with a complete CLI-based training and evaluation workflow, now extended with a working Linux kernel-space inference prototype.

The project has progressed through:

```text
Perceptron Fundamentals
        ↓
Optimized Training
        ↓
Iris Classification
        ↓
Convergence Handling
        ↓
Manual Evaluation
        ↓
Model Persistence
        ↓
Experiment Tracking
        ↓
Structured CLI
        ↓
Linux Kernel Module
        ↓
Fixed-Point Kernel Inference
        ↓
Python ↔ Kernel Integration
        [IN PROGRESS]
```

The project began by answering:

> **How does a perceptron actually learn?**

The current systems stage extends that question to:

> **How can lightweight ML inference move from a high-level Python workflow toward lower-level operating-system execution?**
