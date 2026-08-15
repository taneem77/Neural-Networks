# benchmark.py
# Runs comprehensive benchmarking across multiple datasets

import pandas as pd
import numpy as np

from sklearn.linear_model import Perceptron

from datasets import load_all_datasets
from metrics import evaluate
from visualizer import (
    plot_convergence,
    plot_confusion,
    plot_boundary,
    plot_benchmark,
    plot_sklearn
)

from optimised_perceptron import OptimisedPerceptron


def run_benchmark():

    print("\n" + "="*60)
    print(" PERCEPTRON MULTI-DATASET BENCHMARK ")
    print("="*60)

    datasets = load_all_datasets()

    results = []

    for name, data in datasets.items():

        print(f"\nRunning {name}...")

        # -----------------------------
        # Logical datasets
        # -----------------------------

        if data[2] is None:

            X = data[0]
            y = data[1]

            model = OptimisedPerceptron(
                lr=0.1,
                epochs=50
            )

            model.fit(X, y)

            pred = model.predict(X)

            metrics = evaluate(y, pred)

            sklearn_acc = np.nan

            plot_boundary(model, X, y, name)

        # -----------------------------
        # Real datasets
        # -----------------------------

        else:

            X_train, X_test, y_train, y_test = data

            model = OptimisedPerceptron(
                lr=0.1,
                epochs=100
            )

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            metrics = evaluate(y_test, pred)

            sk = Perceptron(
                max_iter=1000,
                random_state=42
            )

            sk.fit(X_train, y_train)

            sklearn_acc = sk.score(X_test, y_test)

        plot_convergence(model, name)
        plot_confusion(metrics["confusion"], name)

        results.append({
            "Dataset": name,
            "Our Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "Sklearn": sklearn_acc
        })

    df = pd.DataFrame(results)

    plot_benchmark(df)
    plot_sklearn(df)

    df.to_csv(
        "outputs/report.csv",
        index=False
    )

    print("\nResults saved to outputs/report.csv")

    print(df.round(3))