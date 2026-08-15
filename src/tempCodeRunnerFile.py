import numpy as np

class OptimisedPerceptron:
    def __init__(self, lr=0.1, epochs=50):
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
                    self.bias += update        
                    errors += 1                 # Record the mistake count
                    
           
            self.errors_per_epoch.append(errors)
            
          
            # If errors reach absolute zero, weights are optimal. Stop early!
            if errors == 0:
                break

    def predict_single(self, xi):
        # Linear dot-product combination: z = w · x + b
        activation = np.dot(xi, self.weights) + self.bias
        # Step Activation Function returns 1 for positive boundaries, else 0
        return 1 if activation >= 0.0 else 0

    def predict(self, X):
        return np.array([self.predict_single(xi) for xi in X])

    def accuracy(self, X, y):
        preds = self.predict(X)
        return sum(p == a for p, a in zip(preds, y)) / len(y) * 100
