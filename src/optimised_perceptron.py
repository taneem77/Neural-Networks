# =====================================================================
# DAY 2-3: OPTIMISED PERCEPTRON WITH EARLY CONVERGENCE LOOP
# =====================================================================
# This file optimizes the baseline single-layer perceptron. 
# Instead of running through every single epoch blindly, it implements
# an early-stopping check: if an entire epoch passes with 0 mistakes,
# the weights are perfectly optimized, and the loop terminates early!

import numpy as np

class OptimisedPerceptron:
    def __init__(self, lr=0.1, epochs=50):
        # Initialize basic hyperparameters
        self.lr = lr                  # Learning rate handles weight adjustment step sizes
        self.epochs = epochs          # Maximum number of iterations allowed
        self.weights = None           # Weight array matching the feature count
        self.bias = 0.0               # Bias term adjusts the decision boundary threshold
        self.errors_per_epoch = []    # Keeps track of updates made across each epoch iteration

    def fit(self, X, y):
        """ Trains the perceptron weights using optimization routines. """
        n_features = X.shape[1]
        self.weights = np.zeros(n_features) # Initialize weights to zero matrices
        self.bias = 0.0
        self.errors_per_epoch = []

        for epoch in range(self.epochs):
            errors = 0
            
            # Loop through each individual input observation and its true target label
            for xi, yi in zip(X, y):
                # Calculate the weight correction update score
                update = self.lr * (yi - self.predict_single(xi))
                
                # If prediction doesn't match target, execute backprop weight corrections
                if update != 0.0:
                    self.weights += update * xi # Update input direction array
                    self.bias += update         # Shift threshold vector bias
                    errors += 1                 # Record the mistake count
                    
            # Track chronological optimization history trend parameters
            self.errors_per_epoch.append(errors)
            
            # EARLY CONVERGENCE CHECK: 
            # If errors reach absolute zero, weights are optimal. Stop early!
            if errors == 0:
                break

    def predict_single(self, xi):
        """ Evaluates activation threshold logic for a single row index vector. """
        # Linear dot-product combination: z = w · x + b
        activation = np.dot(xi, self.weights) + self.bias
        # Step Activation Function returns 1 for positive boundaries, else 0
        return 1 if activation >= 0.0 else 0

    def predict(self, X):
        """ Map step function vector operations over a full inference dataset. """
        return np.array([self.predict_single(xi) for xi in X])

    def accuracy(self, X, y):
        """ Calculate standard prediction accuracy scores from zero-scratch arrays. """
        preds = self.predict(X)
        return sum(p == a for p, a in zip(preds, y)) / len(y) * 100