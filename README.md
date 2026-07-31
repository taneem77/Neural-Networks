# Perceptron CLI — From-Scratch ML to Linux Kernel Inference

A single-layer perceptron classifier implemented from scratch with NumPy and progressively developed from basic logic-gate experiments into a complete command-line machine-learning workflow.

The project further extends the Python implementation into **Linux systems programming**, with a loadable kernel module that performs fixed-point perceptron inference inside kernel space.

The project currently supports:

- Perceptron training from scratch
- Training convergence and early stopping
- Iris dataset preprocessing
- Manual evaluation metrics
- Prediction on unseen samples
- Model persistence
- Experiment tracking
- Structured CLI commands
- Linux kernel module compilation and loading
- Fixed-point perceptron inference in kernel space
- User-space ↔ kernel-space communication via a character device

> `scikit-learn` is used for dataset loading, splitting, and preprocessing.  
> The perceptron classifier and evaluation metrics are implemented manually.

---

# Project Progression

```text
Perceptron Fundamentals
        ↓
Training Improvements
        ↓
Iris Data Pipeline
        ↓
Convergence Handling
        ↓
Interactive Training
        ↓
Evaluation & Prediction
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
```

The project began by exploring how a perceptron learns and gradually expanded toward understanding how the same lightweight inference operation can be executed at a lower systems level.

---

# File Overview

| File | Purpose |
|---|---|
| `cli.py` | Main command-line entry point that routes training, prediction, evaluation, history, and model-information commands. |
| Python modules under `src/` | Implement the perceptron, Iris preprocessing, training logic, evaluation, persistence, experiment history, and terminal interface. |
| `src/kernel/perceptron_kernel.c` | Implements the Linux kernel module and performs fixed-point perceptron inference. |
| `src/kernel/perceptron_model.h` | Stores the feature count, fixed-point scale factor, prototype weights, and bias used by the kernel module. |
| `src/kernel/perceptron_kmod.c` | Implements a character device that exposes fixed-point dot-product computation to user space via `ioctl()`. |
| `src/kernel/kernel_bridge.py` | User-space bridge that converts vectors to Q16.16 fixed-point, sends them to `/dev/perceptron_kmod`, and converts the result back to floating-point. |
| `src/kernel/Makefile` | Compiles the kernel sources into loadable `.ko` modules using Kbuild. |
| `src/kernel/README.md` | Contains kernel-specific build and execution documentation. |

---

# Stage 1 — Perceptron Fundamentals

The project started with a perceptron implemented manually using Python and NumPy.

For an input vector:

```text
x = [x1, x2, ..., xn]
```

the perceptron calculates:

```text
z = w · x + b
```

A binary step activation is then applied:

```text
prediction = 1    if z >= 0
prediction = 0    otherwise
```

When a sample is incorrectly classified:

```text
error = actual - predicted

w = w + learning_rate × error × x
b = b + learning_rate × error
```

The first experiments used **AND and OR gates**.

This also demonstrates the main limitation of a single-layer perceptron: it can only learn **linearly separable decision boundaries**.

---

# Stage 2 — Training Improvements

The basic training algorithm was extended to make convergence easier to observe and control.

Added features include:

- Error tracking across epochs
- Accuracy tracking
- Best-weight preservation
- Early stopping
- Learning-rate decay
- Confidence-scaled updates
- Convergence monitoring

## Learning-Rate Decay

```text
current_lr = initial_lr × decay^epoch
```

This allows larger updates during early training while gradually reducing the update size as the model approaches convergence.

## Early Stopping

Instead of always executing the configured maximum number of epochs, training can stop once convergence remains stable.

```text
Maximum epochs: 100
Patience: 3

Perfect epoch
Perfect epoch
Perfect epoch
→ Stop training
```

---

# Stage 3 — Iris Data Pipeline

The model was then moved from synthetic logic-gate examples to the Iris dataset.

The current classification problem is:

```text
setosa vs versicolor
```

The data pipeline performs:

```text
Load Iris Dataset
        ↓
Select Two Classes
        ↓
80/20 Train-Test Split
        ↓
Standardize Features
        ↓
Train Perceptron
```

The four input features are:

```text
sepal length (cm)
sepal width (cm)
petal length (cm)
petal width (cm)
```

Feature standardization is calculated using:

```text
x_scaled = (x - mean) / std
```

Standardization prevents features with larger numerical ranges from disproportionately affecting the perceptron's weight updates.

---

# Training

## Command

```bash
python cli.py train
```

A custom learning rate can also be supplied:

```bash
python cli.py train --lr 0.05
```

## Execution Flow

```text
cli.py
  ↓
Iris preprocessing
  ↓
Perceptron training
  ↓
Convergence tracking
  ↓
Training results
```

The training interface displays:

- Dataset information
- Selected classes
- Train/test split
- Learning rate
- Maximum epochs
- Learning-rate decay
- Early-stopping patience
- Training progress
- Train accuracy
- Test accuracy
- Learning curve
- Sample predictions

## Output

![Training Interface](assets/train.png)

---

# Evaluation

## Command

```bash
python cli.py eval --verbose
```

## Execution Flow

```text
cli.py
  ↓
Load model and evaluation data
  ↓
Generate predictions
  ↓
Calculate metrics manually
  ↓
Display evaluation report
```

## Output

![Evaluation Metrics](assets/eval.png)

The evaluation module calculates the confusion matrix manually:

```text
TP — True Positive
TN — True Negative
FP — False Positive
FN — False Negative
```

These values are then used to calculate:

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

The metrics are implemented manually rather than using `sklearn.metrics`.

---

# Prediction

New Iris samples can be classified directly from the command line.

## Command

```bash
python cli.py predict --features 6.3 2.9 4.7 1.6
```

The four values represent:

```text
sepal length
sepal width
petal length
petal width
```

## Execution Flow

```text
cli.py
  ↓
Load saved model
  ↓
Load preprocessing statistics
  ↓
Standardize new sample
  ↓
Calculate w · x + b
  ↓
Apply step activation
  ↓
Return prediction
```

The CLI also displays the **prediction margin**, representing the sample's distance from the perceptron decision boundary.

## Output

![Prediction](assets/predict.png)

---

# Model Persistence

A trained model can be stored and reused instead of retraining before every prediction.

The saved model contains:

- Learned weights
- Bias
- Feature means
- Feature standard deviations
- Model metadata

During training:

```text
x_scaled = (x - training_mean) / training_std
```

During prediction:

```text
new_x_scaled = (new_x - training_mean) / training_std
```

Using the original training statistics ensures that new samples are represented in the same feature space as the training data.

## Inspect Saved Model

```bash
python cli.py info
```

---

# Experiment History

Training experiments can be recorded and compared across different hyperparameter configurations.

## Commands

```bash
python cli.py history --log --lr 0.1
python cli.py history --log --lr 0.05
python cli.py history --log --lr 0.01
```

View recorded experiments:

```bash
python cli.py history
```

Each experiment can record:

- Timestamp
- Learning rate
- Epoch configuration
- Epochs completed
- Test accuracy
- Learned weights

## Output

![Training History](assets/history.png)

---

# Structured CLI

Instead of running separate scripts manually, the project exposes the main workflow through CLI subcommands.

```bash
python cli.py --help
```

## CLI Commands

| Command | Purpose | Output |
|---|---|---|
| `python cli.py train` | Train the perceptron | Training progress and accuracy |
| `python cli.py predict --features ...` | Classify a new sample | Predicted class and decision margin |
| `python cli.py eval --verbose` | Evaluate the trained model | Confusion matrix and evaluation metrics |
| `python cli.py history` | View previous experiments | Recorded training runs |
| `python cli.py info` | Inspect the saved model | Model parameters and metadata |
| `python cli.py --help` | Display CLI documentation | Commands and arguments |

## Output

![CLI Commands](assets/cli_help.png)

---

# Linux Kernel Integration

The latest stage extends the project from a Python ML workflow into Linux systems programming.

The responsibilities are separated as:

```text
USER SPACE

Training
Preprocessing
Evaluation
Model Parameters

      ↓

KERNEL SPACE

Lightweight Inference
```

Training remains in Python.

The kernel prototype focuses on executing the fundamental perceptron inference operation:

```text
z = w · x + b
```

followed by:

```text
prediction = 1 if z >= 0
prediction = 0 otherwise
```

---

# Kernel Files

The kernel implementation is located inside:

```text
src/kernel/
├── perceptron_kernel.c
├── perceptron_model.h
├── perceptron_kmod.c
├── kernel_bridge.py
├── Makefile
└── README.md
```

## `perceptron_kernel.c`

Implements:

- Kernel module initialization
- Kernel module cleanup
- Weighted-sum calculation
- Step activation
- Demonstration inference
- Kernel logging using `pr_info`

The core inference operation follows the same mathematical idea as the Python perceptron:

```text
Input
  ↓
Weighted Sum
  ↓
w · x + b
  ↓
Step Activation
  ↓
Prediction
```

## `perceptron_model.h`

Contains:

- Number of input features
- Fixed-point scaling factor
- Prototype weights
- Prototype bias

The parameters are currently **hardcoded prototype values**.

They are used to verify that the kernel inference path works correctly before automatically transferring trained parameters from Python.

## `Makefile`

Uses the Linux kernel build system to compile the kernel implementation as an external loadable module.

```text
perceptron_kernel.c
        ↓
      Kbuild
        ↓
perceptron_kernel.ko
```

---

# Fixed-Point Arithmetic

The Python implementation uses floating-point NumPy operations.

The kernel prototype instead represents model values using scaled integers.

The current scaling factor is:

```text
SCALE_FACTOR = 1000
```

For example:

```text
 1.0   →  1000
 0.5   →   500
-0.75  →  -750
```

Conceptually:

```text
floating-point value
        ↓
multiply by 1000
        ↓
integer representation
```

This allows the initial kernel inference implementation to use integer arithmetic.

The current weights and bias inside `perceptron_model.h` are not yet the parameters learned by the Python model.

Automatic model export is part of the next integration stage.

---

# Building the Kernel Module

The prototype was tested using:

```text
Ubuntu Linux
Kernel: 6.2.0-39-generic
Architecture: x86_64
```

Move into the kernel directory:

```bash
cd src/kernel
```

Compile the module:

```bash
make
```

## Files Involved

```text
Makefile
    ↓
perceptron_kernel.c
    +
perceptron_model.h
    ↓
Linux Kernel Build System
    ↓
perceptron_kernel.ko
```

The main generated file is:

```text
perceptron_kernel.ko
```

`.ko` represents a **kernel object**, which can be dynamically loaded into the running Linux kernel.

## Output

![Kernel Module Build](assets/kernel_build.png)

---

# Running Kernel-Space Inference

Once the module has been compiled, it can be loaded into the running kernel.

## Load the Module

```bash
sudo insmod perceptron_kernel.ko
```

`insmod` inserts the compiled kernel object into the running Linux kernel.

---

## Verify the Module

```bash
lsmod | grep perceptron
```

`lsmod` displays currently loaded kernel modules.

Piping the result through:

```bash
grep perceptron
```

filters the output to the perceptron module.

---

## View Inference Output

```bash
sudo dmesg | grep perceptron
```

The kernel module uses:

```c
pr_info(...)
```

to write information to the Linux kernel log.

`dmesg` is therefore used to read the inference output produced by the module.

The current demonstration produces:

```text
Input       = [500, -200, 1100, 800]
Activation  = -2350000
Prediction  = 0
```

Since:

```text
activation < 0
```

the binary step activation produces:

```text
prediction = 0
```

## Output

![Kernel Perceptron Inference](assets/kernel_inference.png)

The complete execution path is:

```text
perceptron_model.h
        ↓
Prototype Weights + Bias
        ↓
perceptron_kernel.c
        ↓
Weighted Sum
        ↓
Step Activation
        ↓
pr_info()
        ↓
Linux Kernel Log
        ↓
dmesg
        ↓
Visible Prediction
```

---

## Unload the Module

After execution:

```bash
sudo rmmod perceptron_kernel
```

This removes the perceptron module from the running kernel.

The full kernel workflow is therefore:

```text
make
  ↓
perceptron_kernel.ko
  ↓
insmod
  ↓
Kernel Inference
  ↓
dmesg
  ↓
rmmod
```

---

# Python ↔ Kernel Communication

The project has now been extended beyond standalone kernel inference by introducing a **user-space ↔ kernel-space communication layer**.

A new kernel module, `perceptron_kmod`, exposes a **character device** that allows user-space applications to request fixed-point computations using the Linux `ioctl()` interface.

The communication bridge is implemented in:

```text
src/kernel/kernel_bridge.py
```

Its responsibilities are:

- Convert floating-point vectors into **Q16.16 fixed-point integers**
- Open the `/dev/perceptron_kmod` device
- Send the vectors to the kernel using `ioctl()`
- Receive the computed dot product
- Convert the fixed-point result back into floating-point

---

## Kernel Communication Workflow

```text
Python Perceptron
        │
        ▼
kernel_bridge.py
        │
        ▼
Convert Float → Q16.16
        │
        ▼
ioctl()
        │
════════ USER / KERNEL BOUNDARY ════════
        │
        ▼
perceptron_kmod
        │
        ▼
Fixed-Point Dot Product
        │
        ▼
Return Activation
        │
════════ USER / KERNEL BOUNDARY ════════
        │
        ▼
Convert Q16.16 → Float
        │
        ▼
Prediction in Python
```

---

## Building the Communication Module

Compile both kernel modules:

```bash
make
```

This generates:

```text
perceptron_kernel.ko
perceptron_kmod.ko
```

Load the communication module:

```bash
sudo insmod perceptron_kmod.ko
```

Verify that it is loaded:

```bash
lsmod | grep perceptron
```

Check the kernel log:

```bash
sudo dmesg | tail -20
```

Verify the character device:

```bash
ls /dev/perceptron_kmod
```

### Output

![Kernel Communication Module](assets/kernel_module_loaded.png)

The screenshot above demonstrates:

- Successful loading of `perceptron_kmod`
- Registration of the character device
- Creation of `/dev/perceptron_kmod`
- Successful initialization inside kernel space

---

## Executing a Kernel Computation from Python

Run:

```bash
python3 kernel_bridge.py
```

### Output

![Kernel Bridge Output](assets/kernel_bridge_output.png)

Example:

```text
kernel-computed dot product: -0.79998779296875
expected (python):           -0.8
```

The slight numerical difference is expected because the kernel performs the computation using **Q16.16 fixed-point arithmetic**, while Python performs the same calculation using floating-point arithmetic.

This demonstrates successful communication between Python and the Linux kernel while preserving numerical accuracy within the limits of fixed-point representation.

---

# Current System Architecture

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
                            ▼
                  Fixed-Point Export
                            │
                            ▼
                     kernel_bridge.py
                            │
                            ▼
                        ioctl()

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

The kernel inference path and the user-space ↔ kernel-space communication bridge are both working.

The remaining step is exporting the trained Python model's actual weights and bias into the kernel representation automatically, instead of using hardcoded prototype values.

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
│   ├── kernel_inference.png
│   ├── kernel_module_loaded.png
│   └── kernel_bridge_output.png
│
├── src/
│   ├── Python perceptron and CLI modules
│   │
│   └── kernel/
│       ├── perceptron_kernel.c
│       ├── perceptron_model.h
│       ├── perceptron_kmod.c
│       ├── kernel_bridge.py
│       ├── Makefile
│       └── README.md
│
├── cli.py
├── .gitignore
└── README.md
```

---

# Command → File → Output Reference

| Command | Main Component | Result |
|---|---|---|
| `python cli.py --help` | `cli.py` | Displays available CLI commands |
| `python cli.py train` | CLI + training/perceptron modules | Trains the model and displays convergence information |
| `python cli.py eval --verbose` | CLI + evaluation module | Displays confusion matrix and manually calculated metrics |
| `python cli.py predict --features ...` | CLI + saved model | Standardizes input and produces a prediction |
| `python cli.py info` | CLI + model persistence | Displays saved model information |
| `python cli.py history` | CLI + experiment history | Displays recorded training experiments |
| `make` | `Makefile` + kernel source | Generates `perceptron_kernel.ko` and `perceptron_kmod.ko` |
| `sudo insmod perceptron_kernel.ko` | `perceptron_kernel.ko` | Loads the module and executes initialization/inference |
| `sudo insmod perceptron_kmod.ko` | `perceptron_kmod.ko` | Loads the communication module and registers the character device |
| `lsmod \| grep perceptron` | Linux module system | Confirms that the module is loaded |
| `sudo dmesg \| grep perceptron` | `pr_info()` output | Displays kernel-space inference |
| `python3 kernel_bridge.py` | `kernel_bridge.py` | Sends a vector to the kernel via `ioctl()` and prints the computed result |
| `sudo rmmod perceptron_kernel` | Linux module system | Removes the inference module |
| `sudo rmmod perceptron_kmod` | Linux module system | Removes the communication module |

---

# Key Concepts Covered

| Stage | Concepts |
|---|---|
| Perceptron fundamentals | Weights, bias, weighted sum, activation, learning rule |
| Logic-gate experiments | Linear separability |
| Training improvements | Convergence, early stopping, learning-rate decay, best weights |
| Iris pipeline | Dataset filtering, train/test split, feature standardization |
| Evaluation | TP, TN, FP, FN, accuracy, precision, recall, F1 |
| Prediction | Preprocessing unseen samples and decision margins |
| Persistence | Saving model parameters and scaler statistics |
| Experiment tracking | Comparing hyperparameter configurations |
| CLI | Separating training, evaluation, prediction, history, and inspection |
| Kernel modules | Compilation, insertion, kernel logging, and removal |
| Fixed-point inference | Representing floating-point model values as scaled integers |
| Kernel communication | Character devices, `ioctl()`, user/kernel data transfer |

---

# Current Limitations

The current classifier is a **single-layer perceptron**, so its decision boundary is linear.

A classic example of a problem it cannot solve is XOR:

```text
0 XOR 0 → 0
0 XOR 1 → 1
1 XOR 0 → 1
1 XOR 1 → 0
```

No single linear boundary can separate the XOR classes.

The kernel implementation is also currently an initial prototype.

At this stage:

- Training remains in Python
- Preprocessing remains in user space
- Kernel weights and bias in `perceptron_kernel.c` are hardcoded prototype values
- `perceptron_kmod` computes dot products passed to it, but the perceptron's own weights are not yet passed through automatically
- Trained Python parameters are not yet exported automatically into either kernel module

---

# Tech Stack

| Area | Technologies |
|---|---|
| Machine Learning | Python, NumPy |
| Dataset & Preprocessing | scikit-learn |
| CLI | argparse |
| Systems Programming | C |
| Operating System | Linux / Ubuntu |
| Kernel Development | Linux Kernel Modules, Kbuild, character devices, ioctl |
| Version Control | Git, GitHub |

---

# Example Workflow

## Python Workflow

```bash
# View available commands
python cli.py --help

# Train the model
python cli.py train

# Evaluate the model
python cli.py eval --verbose

# Predict an unseen sample
python cli.py predict --features 6.3 2.9 4.7 1.6

# Inspect the saved model
python cli.py info

# View experiment history
python cli.py history
```

## Kernel Workflow

```bash
# Enter kernel directory
cd src/kernel

# Compile the modules
make

# Load the inference module
sudo insmod perceptron_kernel.ko

# Verify that it is loaded
lsmod | grep perceptron

# View kernel inference output
sudo dmesg | grep perceptron

# Load the communication module
sudo insmod perceptron_kmod.ko

# Verify the character device
ls /dev/perceptron_kmod

# Run a computation from Python
python3 kernel_bridge.py

# Unload the modules
sudo rmmod perceptron_kmod
sudo rmmod perceptron_kernel
```

---

# Status

**Current stage:** Single-layer perceptron with a complete CLI-based training, prediction, and evaluation workflow, extended with a working Linux kernel-space inference prototype and a functioning user-space ↔ kernel-space communication bridge.

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
Python ↔ Kernel Communication
        ↓
Trained-Model Export
        [IN PROGRESS]
```

The project began with:

> **How does a perceptron actually learn?**

and currently extends that idea toward:

> **How can lightweight ML inference move from a Python workflow toward lower-level operating-system execution, with a working communication path between the two?**
