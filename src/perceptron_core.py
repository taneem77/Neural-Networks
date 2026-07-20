import numpy as np #maths library

x = np.array([1.0, 2.0, 3.0])
w = np.zeros(3)           # initialise weights to zero
print(np.dot(w, x))    

def step(z):
    return 1 if z >= 0 else 0 #perceptron makes binary decision so 0 and 1 

class Dog:
    def __init__(self, name):#constructor, __init__ is called automatically by python when an object is created (its to understand and assign attricbutes to the object)
        self.name = name #Put this information inside the dog object, i.e if this wasnt there, thered be no memory stored 

    def speak(self): #self is to keep object data, more specifically the object ur currently working with
        print(f"{self.name} says woof") #brune says woof

d = Dog("Bruno")
d.speak()

class Perceptron:
    #need to understand : weights, bias, how many times to epoch for learning 
    #Inputs: Multiply by weights, Add bias, z, Threshold, Prediction
    #Every perceptron shld carry the same i.e weights bias and learning rate and epochs
    def __init__(self, lr=0.1, epochs=10): #this is to learn and remember the object (why? : first of all you need to know the prev value of the perceptron and then secondly one u get the value or output u want from a perceptron u want it to rememeber so that it can use on a diff data right, so you need to rem the weight of the perceptron and decision boundary as well which is given by bias)
        #New perceptron created: Give it learning rate,  epochs, Create empty weights and bias
        #learning and learning rate is so that the perceptrons makes small small changes everytime it encounters an error
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y): #x here is the input training data, y is the correct correlation for x as input data
        #important point to note: fit doesnt return anything, what it does is it stores it in perceptron memory
        #this is repeated, the number equal to the value of epoch in the code
        self.weights = np.zeros(X.shape[1]) 
        #each feature is given a weight, its usually shape(number of training data, number of features), but here its x.shape(1) i.e number of coumns/features
        for epoch in range(self.epochs):
            for xi, yi in zip(X, y): #basically the current input and its corresponding output 
                #point to note here: ur second epoch starts w updated weight, i.e 1st epoch weight is 0,0 and then u get an output say 0.1 so ur second epoch starts at 0.1 
                z = np.dot(self.weights, xi) + self.bias 
                pred = 1 if z >= 0 else 0 # this is core, basically we understand that this is single layer, as in the decision it makes are just yes or no etc, so here what we are doing is just making it into two classes 1 and 0, and acc to the result its divided into yes or no (1/0)
                error = yi - pred 
                self.weights += self.lr * error * xi #Correction=Learning Rate×Error×Input
                self.bias    += self.lr * error #since it doesnt have feature attached to it 

    def predict(self, X): #this is to use the training data
        return [1 if np.dot(self.weights, xi) + self.bias >= 0 else 0 for xi in X]
    

    # (or to understand we can write predcition like this )predictions = []

#for xi in X:

   # z = np.dot(self.weights, xi) + self.bias

    #if z >= 0:
      #  predictions.append(1)
    #else:
      #  predictions.append(0)

#return predictions
    #fit was to learn and predict is to use what was learnt 

 #and gate 
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float) # to store the numbers as decimal
y = np.array([0, 0, 0, 1])

p = Perceptron()
p.fit(X, y)
pred = p.predict(X)

print(" AND Gate ")
print("Expected :", y)
print("Predicted:", pred)
print("Correct? :", np.array_equal(pred, y))
print("Weights  :", p.weights)
print("Bias     :", p.bias)
print()

#or gate
X1 = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float) # to store the numbers as decimal
y1 = np.array([0, 1, 1, 1])

p1 = Perceptron()
p1.fit(X1, y1)
pred = p1.predict(X1)

print("OR Gate")
print("Expected :", y1)
print("Predicted:", pred)
print("Correct? :", np.array_equal(pred, y1))
print("Weights  :", p1.weights)
print("Bias     :", p1.bias)
print()

# simple test : 
X2 = np.array([[7],[3],[88],[2]], dtype=float) # to store the numbers as decimal
y2 = np.array([0, 1, 1, 1])

p2 = Perceptron()
p2.fit(X2, y2)
pred = p2.predict(X2)

print(" Simple Test")
print("Expected :", y2)
print("Predicted:", pred)
print("Correct? :", np.array_equal(pred, y2))
print("Weights  :", p2.weights)
print("Bias     :", p2.bias)
print()



#theres this thing called convergence, which is not implemented yet
#since this is single layer only lineraly seperable data can and will work , therefor the simple test may or may not work acc to data 
#it requires manually choosing the learning rate thats a drawback because depedning on the type of data 0.1 and all could be tooo small 
#it requires manually choosing the number of epochs, this goes hand in hand with the convergence drawback 
#it gives only a hard decision (0 or 1) rather than a confidence score i.e that this result is 90% true 