# Perceptron CLI — A From-Scratch ML Tool

A single-layer perceptron classifier implemented from scratch with NumPy and
developed progressively from basic logic-gate experiments into a complete
command-line machine learning workflow.

The project supports training, evaluation, prediction, model persistence,
experiment tracking, hyperparameter tuning, and an interactive terminal
interface.

The core perceptron algorithm and evaluation metrics are implemented manually
rather than using pre-built ML classifiers or `sklearn.metrics`.

---

## Overview

The goal of this project is to understand what happens inside a perceptron
instead of treating machine learning as a black box.

The project began with a minimal implementation capable of learning simple
logic gates and was gradually extended with convergence handling, real-world
data preprocessing, evaluation metrics, model persistence, experiment logging,
and a structured CLI.

The current classifier operates on a binary subset of the Iris dataset:

```text
setosa vs versicolor
```

---

## Project Progress

### Stage 1 — Perceptron Fundamentals

The project started with a basic perceptron implemented from scratch.

The initial model included:

- Weight initialization
- Bias handling
- Weighted-sum calculation
- Step activation function
- Prediction
- Perceptron learning rule
- Epoch-based training

It was tested on simple problems including AND and OR gates.

These experiments also demonstrated a fundamental limitation of a
single-layer perceptron: it can only learn linearly separable decision
boundaries.

---

### Stage 2 — Training Improvements

The initial implementation was extended to make training more stable,
observable, and efficient.

Added:

- Error tracking across epochs
- Accuracy tracking
- Best-weight preservation
- Early stopping
- Learning-rate decay
- Confidence-scaled updates
- Convergence monitoring

Instead of blindly using the final epoch's parameters, the model can preserve
its best-performing weights and stop training once convergence is detected.

---

### Stage 3 — Iris Data Pipeline

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

Feature standardization is important for perceptron training because features
with larger numerical ranges could otherwise dominate the weight updates.

`scikit-learn` is used for dataset loading, train-test splitting, and
preprocessing. The classifier itself is implemented from scratch.

---

### Stage 4 — Interactive Training Interface

An interactive terminal interface was added to make the training process
visible instead of hiding it behind a single function call.

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

After training, the model can also be retrained interactively with different
hyperparameters.

![Training Interface](assets/train.png)

---

### Stage 5 — Structured CLI

The project was reorganized into a git-style command-line interface with
separate subcommands for different parts of the ML workflow.

```bash
python cli.py train
python cli.py predict --features 5.1 3.5 1.4 0.2
python cli.py eval --verbose
python cli.py history
python cli.py info
```

This separates training, inference, evaluation, model inspection, and
experiment tracking instead of handling everything inside one script.

![CLI Help](assets/cli_help.png)

---

## Training

The model can be trained directly from the command line:

```bash
python cli.py train
```

Training can also be configured using different hyperparameters.

For example:

```bash
python cli.py train --lr 0.05
```

The training process tracks convergence and stores information such as
errors, accuracy, learning rate, and learned parameters.

Early stopping prevents unnecessary epochs once the classifier has
converged.

---

## Evaluation

Model evaluation is available through:

```bash
python cli.py eval --verbose
```

The evaluation module calculates the metrics manually from model predictions
rather than using `sklearn.metrics`.

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

Verbose mode also explains what each metric represents rather than only
printing the numerical value.

![Evaluation Metrics](assets/eval.png)

### Confusion Matrix

For binary classification, predictions are separated into:

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

Accuracy measures the fraction of all predictions that were correct.

### Precision

```text
Precision = TP / (TP + FP)
```

Precision answers:

> When the model predicts the positive class, how often is it correct?

### Recall

```text
Recall = TP / (TP + FN)
```

Recall answers:

> Of all actual positive samples, how many did the model identify?

### F1 Score

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 combines precision and recall into a single metric.

---

## Prediction

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

Before inference, the new sample is standardized using the same normalization
statistics learned from the training data.

The CLI also displays a prediction margin, providing an indication of how far
the sample lies from the perceptron's decision boundary.

![Prediction](assets/predict.png)

---

## Model Persistence

A trained model can be stored and reused instead of being retrained for every
prediction.

The saved model includes:

- Learned weights
- Bias
- Feature means
- Feature standard deviations
- Model metadata

Saving preprocessing statistics alongside the model is important.

During training, the input features are standardized as:

```text
x_scaled = (x - mean) / std
```

If a new prediction were normalized using different statistics, the input
would exist in a different feature space from the one the perceptron was
trained on.

For this reason, the original training mean and standard deviation are saved
and reused during inference.

---

## Experiment History

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

This makes it possible to compare how different hyperparameter choices affect
training behavior.

![Training History](assets/history.png)

---

## CLI Commands

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

## Perceptron Implementation

For an input vector:

```text
x = [x1, x2, ..., xn]
```

the perceptron first calculates the weighted sum:

```text
z = w · x + b
```

where:

```text
w = learned weights
b = bias
```

A step activation function converts the result into a binary prediction:

```text
prediction = 1    if z > 0
prediction = 0    otherwise
```

When a sample is misclassified, the parameters are updated using the
perceptron learning rule.

```text
error = actual - predicted

w = w + learning_rate × error × x
b = b + learning_rate × error
```

The optimized implementation additionally experiments with
confidence-scaled updates and learning-rate decay.

---

## Learning Rate Decay

Instead of keeping the learning rate constant throughout training, the
convergent perceptron can gradually reduce it.

Conceptually:

```text
current_lr = initial_lr × decay^epoch
```

This allows larger updates early in training while reducing the update size
as the model approaches convergence.

The terminal interface tracks this decay across epochs.

---

## Early Stopping

The model does not necessarily need to run for every configured epoch.

If it achieves zero classification errors for the configured patience period,
training can stop early.

For example:

```text
Maximum epochs: 100
Patience: 3

Perfect epoch
Perfect epoch
Perfect epoch
→ Stop training
```

This avoids unnecessary computation once the model has converged.

---

## Project Architecture

The overall ML workflow is:

```text
                     Raw Iris Dataset
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
                    Experiment History
```

---

## Implementation Highlights

### From-scratch classifier

The perceptron learning algorithm is implemented manually using NumPy rather
than a pre-built classifier.

### Manual evaluation metrics

Accuracy, precision, recall, F1, and the confusion matrix are derived directly
from predictions.

### Consistent preprocessing

Training normalization statistics are stored and reused during inference,
preventing preprocessing differences between training and prediction.

### Convergence tracking

Errors, accuracy, and learning rate can be tracked across epochs.

### Experiment logging

Multiple runs can be recorded and compared instead of losing previous
hyperparameter experiments.

### Command-line workflow

Training, evaluation, prediction, history, and model information are exposed
through separate CLI subcommands.

---

## Current Limitations

The current model is still a **single-layer perceptron**.

Its decision boundary is therefore linear.

This makes it suitable for linearly separable classification problems such as
the selected Iris classes, but it cannot learn non-linearly separable
relationships.

A classic example is XOR:

```text
0 XOR 0 → 0
0 XOR 1 → 1
1 XOR 0 → 1
1 XOR 1 → 0
```

No single straight decision boundary can separate the two XOR classes.

Solving this requires introducing hidden layers and non-linear activation
functions.

---

## What's Next

The next stage of the project is to move beyond a single-layer classifier.

Planned extensions:

- Build a multi-layer perceptron from scratch
- Implement forward propagation
- Implement backpropagation manually
- Add non-linear activation functions
- Train on non-linearly separable problems such as XOR
- Add automated unit tests for model behavior
- Add unit tests for manually implemented evaluation metrics

---

## Tech Stack

- **Python**
- **NumPy** — numerical operations and perceptron implementation
- **scikit-learn** — Iris dataset loading, train/test splitting, and preprocessing
- **argparse** — command-line interface
- **PowerShell / Terminal** — CLI interaction

---

## Example Workflow

A complete workflow can be run from the terminal:

```bash
# Train the model
python cli.py train

# Inspect model information
python cli.py info

# Make a prediction
python cli.py predict --features 6.3 2.9 4.7 1.6

# Evaluate the trained model
python cli.py eval --verbose

# View experiment history
python cli.py history
```

---

## Status

**Current stage:** Single-layer perceptron with a complete CLI-based
training and evaluation workflow.

The project has progressed from basic perceptron experiments to a reusable
machine-learning tool with preprocessing, evaluation, persistence, inference,
and experiment tracking.
