#Learning rate decay + patience convergence + comparison of all models
#so this gile basic has convergentperceptron and also compares all the three algorithms written
#what makes this diff cus it has patience based convergence 

import numpy as np #Mathematicla operations
import importlib #importing other perceptron algos dynamically
import data_pipeline #importing that file
from persist import save_model


# Dynamic imports to load older configurations safely
day1_2 = importlib.import_module("perceptron_core")
BasePerceptron = day1_2.Perceptron

day2_3 = importlib.import_module("optimised_perceptron")
OptimisedPerceptron = day2_3.OptimisedPerceptron

# Bring in the data pipeline from day4 cleanly
build_pipeline = data_pipeline.build_pipeline #Local ref to the function build pipeline


class ConvergentPerceptron:
    def __init__(self, lr=0.1, epochs=100, decay=0.99, patience=3): #decay is the new concept and so is patience
        #patience is basically isntead of getting 1 stable output and declaring it , u wait for three consecutive outputs and then declare stablility
        self.lr       = lr
        self.epochs   = epochs
        self.decay    = decay      
        self.patience = patience   

        self.weights  = None
        self.bias     = 0.0 #weights dont exist yet u start at 0 

        self.history = { #this is directory of sorts to trakc what happens in the training process 
            'errors':   [],   
            'accuracy': [],   
            'lr':       [],   
        } #itll show u how training changed overtime instead of sending one data only 

    def _current_lr(self, epoch):
        return self.lr * (self.decay ** epoch) #I'm getting closer to a solution, so make progressively smaller adjustments

    def fit(self, X, y): #Train the perceptron.
        n_features = X.shape[1] #Train the perceptron ex: X.Shape(80,4)-> 80 samples, 4 features each
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.history = {'errors': [], 'accuracy': [], 'lr': []}

        best_weights  = np.zeros(n_features)# so every feature starts of with a 0
        best_bias     = 0.0
        best_accuracy = -1.0 #cus accuracy cant be below 0 u can take that as a starting point
        consecutive_perfect = 0   # new concpet: this is the patience counter 

        for epoch in range(self.epochs):
            lr_now       = self._current_lr(epoch)
            epoch_errors = 0

            for xi, yi in zip(X, y):
                z     = np.dot(self.weights, xi) + self.bias
                pred  = 1 if z >= 0 else 0
                error = yi - pred

                if error != 0:
                    epoch_errors += 1
                    confidence_scale = min(1.0 + abs(z), 3.0)
                    self.weights += lr_now * error * confidence_scale * xi
                    self.bias    += lr_now * error * confidence_scale

            preds    = self.predict(X)
            accuracy = sum(p == a for p, a in zip(preds, y)) / len(y)

            self.history['errors'].append(epoch_errors)
            self.history['accuracy'].append(accuracy)
            self.history['lr'].append(lr_now)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights  = self.weights.copy()
                best_bias     = self.bias

            if epoch_errors == 0:
                consecutive_perfect += 1
                if consecutive_perfect >= self.patience:
                    break
            else:
                consecutive_perfect = 0   

        self.weights = best_weights
        self.bias    = best_bias

    def predict(self, X):
        return [1 if np.dot(self.weights, xi) + self.bias >= 0 else 0 for xi in X]

    def accuracy(self, X, y):
        preds = self.predict(X)
        return sum(p == a for p, a in zip(preds, y)) / len(y) * 100


def compare_all(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print(" COMPARISON: Base vs Optimised vs Convergent")
    print("="*60)

    results = {}

    # logic 1
    base = BasePerceptron(lr=0.1, epochs=50)
    base.fit(X_train, y_train)
    base_preds = base.predict(X_test)
    base_acc = sum(p == a for p, a in zip(base_preds, y_test)) / len(y_test) * 100 #predicted labels vs actual test labels
    results['Base'] = {'test_acc': base_acc, 'epochs': 50}

    # logic 2
    opt = OptimisedPerceptron(lr=0.1, epochs=50)
    opt.fit(X_train, y_train)
    opt_acc = opt.accuracy(X_test, y_test)
    results['Optimised'] = {'test_acc': opt_acc, 'epochs': len(opt.errors_per_epoch)} 

    # logic 3
    conv = ConvergentPerceptron(lr=0.1, epochs=100, decay=0.99, patience=3)
    conv.fit(X_train, y_train)
    conv_acc = conv.accuracy(X_test, y_test)
    results['Convergent'] = {'test_acc': conv_acc, 'epochs': len(conv.history['errors'])}

    print("\n SUMMARY TABLE ")
    print(f"  {'Version':<14}  {'Test Acc':>10}  {'Epochs Used':>12}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*12}")
    for name, r in results.items():
        print(f"  {name:<14}  {r['test_acc']:>9.1f}%  {r['epochs']:>12}")

    # save the best convergent model so prediction + kernel use identical weights
    save_model(conv, X_train)

    return conv
