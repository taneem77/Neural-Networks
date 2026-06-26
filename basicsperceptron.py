import numpy as np

# 1. Dataset (XOR Gate)
# Inputs: 4 states of two binary switches
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
# Expected Outputs: 1 if inputs are different, 0 if they are the same
y = np.array([[0], [1], [1], [0]])

# 2. Network Architecture
input_size = 2
hidden_size = 3  # Hidden layer nodes to capture non-linear boundary, can be 2 and so on
output_size = 1

# Seed for reproducibility
np.random.seed(42) # this is so the random value is generated from 42 as a boundary

# Initialize weights and biases randomly: there are 2 weight matrix and 2 biases 
W1 = np.random.uniform(size=(input_size, hidden_size))
b1 = np.zeros((1, hidden_size))
W2 = np.random.uniform(size=(hidden_size, output_size))
b2 = np.zeros((1, output_size))

# 3. Activation Function & Derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 4. Training Loop 
epochs = 20000
lr = 0.1  # Learning rate

for epoch in range(epochs):
    # --- FORWARD PASS ---
    # Layer 1 (Hidden)
    zh = np.dot(X, W1) + b1
    ah = sigmoid(zh)
    
    # Layer 2 (Output)
    zo = np.dot(ah, W2) + b2
    ao = sigmoid(zo)
    
    # --- BACKPROPAGATION ---
    # Calculate error at output
    error = y - ao
    
    # Gradient at output layer
    d_output = error * sigmoid_derivative(ao)
    
    # Calculate error at hidden layer (backpropagate the output gradient through W2)
    error_hidden = d_output.dot(W2.T)
    
    # Gradient at hidden layer
    d_hidden = error_hidden * sigmoid_derivative(ah)
    
    # --- WEIGHT UPDATES ---
    W2 += ah.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

# 5. Verify Results
print("Predictions after training:")
print(np.round(ao, 3))