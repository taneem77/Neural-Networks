
#Instead of manually checking w datsets we use machine learning datasets

import numpy as np
#sklearn is a mchine learning library 
from sklearn.datasets import load_iris #flower data
from sklearn.model_selection import train_test_split# training data and testing data is diff 
from sklearn.preprocessing import StandardScaler #scaling feature


def build_pipeline(verbose=False): #this is a new function, which means everytime we call a function a entire datasrt is created for us
    #Loads Iris dataset, filters to binary classes, and scales the features.
    iris = load_iris()
    #basically the iris dataset has 150 flowers and each flower has 4 features -length and width of sepal & length and width of petal
    #three species also: setosa, versiclor, virginica
    
    # Filter to only setosa (0) and versicolor (1) for binary classification
    #basically setosa-0; versicolor-1 and virginica-2
    binary_idx = iris.target < 2 #since perceptron can do only 1 and 0, theres no 2 we can remove virginica
    X = iris.data[binary_idx] #row for mask being true i.e 0 and 1
    y = iris.target[binary_idx]#colum for the same

    # Split into Train (80%) and Test (20%), i.e there are suppose 100 flowers now so 80 will train and 20 will learn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (Mean = 0, Std = 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)#this finds mean and standard deviation of feature
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, iris.feature_names, iris.target_names
"""X_train → Training inputs
X_test → Testing inputs
y_train → Training labels
y_test → Testing labels
iris.feature_names → ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
iris.target_names → ['setosa', 'versicolor', 'virginica']"""


'''Load Iris Dataset (150 flowers)
            │
            ▼
Keep only Setosa (0) and Versicolor (1)
            │
            ▼
100 flowers remain (binary classification)
            │
            ▼
Split data
80% → Training
20% → Testing
            │
            ▼
Calculate mean and standard deviation from training data
            │
            ▼
Scale training data
            │
            ▼
Scale test data using the SAME values
            │
            ▼
Return everything needed for training and evaluation'''