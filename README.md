
# Neural Networks & Perceptrons: Learning Report

[cite_start]A concise conceptual breakdown of the foundational architectures of Artificial Neural Networks (ANNs), tracking the core principles from single-layer perceptrons to multi-layer processing[cite: 1, 2].

---

## Architectural Overview

### Artificial vs. Biological Neurons
* [cite_start]**Biological Neuron:** Receives signals via dendrites, processes them in the cell body, and transmits outputs through the axon, learning by changing synaptic strength[cite: 13, 14].
* [cite_start]**Artificial Neuron:** Receives numerical inputs, multiplies them by weights, computes a weighted sum, applies an activation function, and produces an output, learning by updating those weights[cite: 15, 16].

### Network Layers
* [cite_start]**Input Layer:** Takes in raw data (e.g., pixels, shape, color)[cite: 9].
* [cite_start]**Hidden Layer:** Learns underlying features and patterns from that data[cite: 10].
* [cite_start]**Output Layer:** Provides the final decision or classification[cite: 11].

---

## Core Machine Learning Concepts

* [cite_start]**Perceptron:** The simplest form of an artificial neural network[cite: 18]. [cite_start]It computes a weighted sum of inputs and checks if the result crosses a threshold to make a binary decision (`1` for yes, `0` for no)[cite: 19, 20].
* [cite_start]**Decision Boundary:** The line or plane that separates classes in the dataset[cite: 32, 33]. 
  * [cite_start]**Single-Layer Perceptron (SLP):** Restricted to a straight line; can only solve linearly separable problems[cite: 24, 35].
  * [cite_start]**Multi-Layer Perceptron (MLP):** Utilizes hidden layers and non-linear activations to map complex or curved boundaries[cite: 26, 28, 36].
* [cite_start]**Weights & Biases:** Weights determine the angle and slope of the decision boundary[cite: 38]. [cite_start]The bias acts as an intercept, shifting the boundary to adjust the network's inherent tendency to fire a positive or negative decision[cite: 39, 40].
* [cite_start]**Activation Function (Sigmoid):** Squashes any real number into a value between `0` and `1`, making it ideal for binary outputs[cite: 42, 43, 44].
* [cite_start]**Partial Data Processing:** Missing values contribute zero to the weighted sum, allowing other weights to compensate via masking techniques[cite: 47, 48, 49].

---

##  Operational Phases

1. [cite_start]**Learning / Training:** Weights and biases are iteratively adjusted using training data to minimize error[cite: 51].
2. [cite_start]**Recall / Inference:** The learned parameters are frozen and used to predict outputs for new, unseen inputs[cite: 52].

```
