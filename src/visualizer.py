# visualizer.py
# Creates publication-style visualizations for the internship report

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# -----------------------------
# Create folders automatically
# -----------------------------

os.makedirs("outputs/confusion", exist_ok=True)
os.makedirs("outputs/convergence", exist_ok=True)
os.makedirs("outputs/boundaries", exist_ok=True)
os.makedirs("outputs/benchmark", exist_ok=True)


# -----------------------------
# Convergence Plot
# -----------------------------

def plot_convergence(model, name):

    plt.figure(figsize=(6,4))

    plt.plot(
        model.history["errors"],
        marker="o",
        linewidth=2
    )

    plt.title(f"{name} Training Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Misclassifications")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"outputs/convergence/{name.replace(' ','_')}.png",
        dpi=300
    )

    plt.close()


# -----------------------------
# Confusion Matrix
# -----------------------------

def plot_confusion(cm, name):

    # FIX: convert Python list -> NumPy array
    cm = np.array(cm)

    fig, ax = plt.subplots(figsize=(5,5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(
        ax=ax,
        colorbar=False,
        cmap="Blues"
    )

    ax.set_title(f"{name} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        f"outputs/confusion/{name.replace(' ','_')}.png",
        dpi=300
    )

    plt.close(fig)


# -----------------------------
# Decision Boundary
# -----------------------------

def plot_boundary(model, X, y, name):

    if X.shape[1] != 2:
        return

    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(5,5))

    plt.contourf(xx, yy, Z, alpha=0.25)

    plt.scatter(
        X[:,0],
        X[:,1],
        c=y,
        edgecolor="black",
        s=55
    )

    plt.title(f"{name} Decision Boundary")

    plt.tight_layout()

    plt.savefig(
        f"outputs/boundaries/{name.replace(' ','_')}.png",
        dpi=300
    )

    plt.close()


# -----------------------------
# Benchmark Accuracy Plot
# -----------------------------

def plot_benchmark(df):

    plt.figure(figsize=(8,4))

    plt.bar(
        df["Dataset"],
        df["Our Accuracy"],
        width=0.6
    )

    plt.ylim(0,1.05)

    plt.ylabel("Accuracy")

    plt.title("Perceptron Accuracy Across Datasets")

    plt.xticks(rotation=25)

    plt.tight_layout()

    plt.savefig(
        "outputs/benchmark/accuracy_comparison.png",
        dpi=300
    )

    plt.close()


# -----------------------------
# Sklearn Comparison
# -----------------------------

def plot_sklearn(df):

    valid = df.dropna()

    x = np.arange(len(valid))

    width = 0.35

    plt.figure(figsize=(8,4))

    plt.bar(
        x-width/2,
        valid["Our Accuracy"],
        width,
        label="Custom"
    )

    plt.bar(
        x+width/2,
        valid["Sklearn"],
        width,
        label="Sklearn"
    )

    plt.xticks(x, valid["Dataset"], rotation=20)

    plt.ylim(0.8,1.05)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/benchmark/sklearn_vs_custom.png",
        dpi=300
    )

    plt.close()