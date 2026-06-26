# Multi-Layer Perceptron (MLP) Implementation for the XOR Problem

This section covers the implementation of a Multi-Layer Perceptron (MLP) designed to solve the classic XOR (Exclusive OR) problem, overcoming the structural limitations of single-layer networks.

---

## The XOR Problem and Single-Layer Limitations

A single-layer perceptron fundamentally fails to resolve the XOR function because the problem is not linearly separable. In an XOR dataset, no single straight line can successfully split the data points into their correct binary classes. 

To overcome this limitation, a Multi-Layer Perceptron utilizes a hidden layer and non-linear activation functions to create a curved or complex decision boundary capable of separating the data.

---

## Network Architecture

The implemented network uses a 3-layer topology to process the binary states:

* **Input Layer:** 2 nodes, representing the binary ON and OFF states of two independent switches.
* **Hidden Layer:** 3 nodes (this value is configurable and can scale from 2 to any arbitrary number of hidden units).
* **Output Layer:** 1 node, representing the final decision of whether the target light bulb is ON or OFF.
* **Parameters:** The network manages two distinct weight matrices ($W_1, W_2$) and two bias vectors ($b_1, b_2$) to pass data across the input-to-hidden and hidden-to-output connections.

---

## Operational Workflow

The implementation executes across the following computational phases:

1. **Random Initialization:** Weights are initialized randomly using a fixed seed. This random setup avoids symmetry errors; if all weights began identical, every hidden neuron would learn the exact same patterns.
2. **Forward Pass:** Input data flows sequentially through the network layer by layer. At each layer, a weighted sum is computed and passed through a Sigmoid activation function to constrain the node values between `0` and `1`.
3. **Error Evaluation:** The difference between the network's prediction and the actual target answer is calculated. This error metric directly drives the training cycle.
4. **Backpropagation:** The computed error is traced backwards through the network from the output layer, to the hidden layer, and finally to the input layer. Gradients are calculated to determine exactly how much each individual weight contributed to the total error.
5. **Convergence:** The training loop is repeated over **20,000 epochs**. This continuous iteration refines the weights and biases based on the errors, allowing the network to converge and correctly predict XOR outputs close to absolute binary `0` and `1`.
