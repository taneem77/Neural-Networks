# Neural Networks & Perceptrons: Learning Report

A concise conceptual breakdown of the foundational architectures of Artificial Neural Networks (ANNs), tracking the core principles from single-layer perceptrons to multi-layer processing.

---

## Architectural Overview

### Artificial vs. Biological Neurons
* **Biological Neuron:** Receives signals via dendrites, processes them in the cell body, and transmits outputs through the axon, learning by changing synaptic strength.
* **Artificial Neuron:** Receives numerical inputs, multiplies them by weights, computes a weighted sum, applies an activation function, and produces an output, learning by updating those weights.

### Network Layers
* **Input Layer:** Takes in raw data (e.g., pixels, shape, color).
* **Hidden Layer:** Learns underlying features and patterns from that data.
* **Output Layer:** Provides the final decision or classification.

---

## Core Machine Learning Concepts

* **Perceptron:** The simplest form of an artificial neural network. It computes a weighted sum of inputs and checks if the result crosses a threshold to make a binary decision (1 for yes, 0 for no).
* **Decision Boundary:** The line or plane that separates classes in the dataset.
  * **Single-Layer Perceptron (SLP):** Restricted to a straight line; can only solve linearly separable problems.
  * **Multi-Layer Perceptron (MLP):** Utilizes hidden layers and non-linear activations to map complex or curved boundaries.
* **Weights & Biases:** Weights determine the angle and slope of the decision boundary. The bias acts as an intercept, shifting the boundary to adjust the network's inherent tendency to fire a positive or negative decision.
* **Activation Function (Sigmoid):** Squashes any real number into a value between 0 and 1, making it ideal for binary outputs.
* **Partial Data Processing:** Missing values contribute zero to the weighted sum, allowing other weights to compensate via masking techniques.

---

## Operational Phases

1. **Learning / Training:** Weights and biases are iteratively adjusted using training data to minimize error.
2. **Recall / Inference:** The learned parameters are frozen and used to predict outputs for new, unseen inputs.
