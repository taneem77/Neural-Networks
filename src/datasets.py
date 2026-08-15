# datasets.py
# Multiple datasets used for benchmarking the perceptron

import numpy as np

from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    load_wine,
    load_digits
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(X, y):
    """Standardise features and create train/test split"""

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def load_all_datasets():

    datasets = {}

    # -----------------------------
    # Logical Gate Datasets
    # -----------------------------

    datasets["AND"] = (
        np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float),
        np.array([0,0,0,1]),
        None
    )

    datasets["OR"] = (
        np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float),
        np.array([0,1,1,1]),
        None
    )

    datasets["NAND"] = (
        np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float),
        np.array([1,1,1,0]),
        None
    )

    datasets["XOR"] = (
        np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float),
        np.array([0,1,1,0]),
        None
    )

    # -----------------------------
    # Iris (Setosa vs Versicolor)
    # -----------------------------

    iris = load_iris()

    mask = iris.target < 2

    datasets["Iris"] = preprocess(
        iris.data[mask],
        iris.target[mask]
    )

    # -----------------------------
    # Breast Cancer
    # -----------------------------

    bc = load_breast_cancer()

    datasets["Breast Cancer"] = preprocess(
        bc.data,
        bc.target
    )

    # -----------------------------
    # Wine (Binary)
    # -----------------------------

    wine = load_wine()

    mask = wine.target < 2

    datasets["Wine"] = preprocess(
        wine.data[mask],
        wine.target[mask]
    )

    # -----------------------------
    # Digits (0 vs 1)
    # -----------------------------

    digits = load_digits()

    mask = (digits.target == 0) | (digits.target == 1)

    datasets["Digits"] = preprocess(
        digits.data[mask],
        digits.target[mask]
    )

    return datasets