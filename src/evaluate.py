import numpy as np
import argparse
from data_pipeline import build_pipeline
from optimised_perceptron import OptimisedPerceptron


def _get_predictions():
    import io, sys
    X_train, X_test, y_train, y_test, _, class_names = build_pipeline(verbose=False)
    model = OptimisedPerceptron(lr=0.1, epochs=100)
    buf = io.StringIO(); sys.stdout = buf
    model.fit(X_train, y_train)
    sys.stdout = sys.__stdout__
    return np.array(model.predict(X_test)), y_test, class_names


# ── confusion matrix ──────────────────────────────────────────────────────────
# not using sklearn.metrics — built from scratch so every number is explainable
#
# what each cell means:
#   TP (true positive)  — predicted versicolor, actually was versicolor   [correct]
#   TN (true negative)  — predicted setosa, actually was setosa           [correct]
#   FP (false positive) — predicted versicolor, actually was setosa       [wrong — false alarm]
#   FN (false negative) — predicted setosa, actually was versicolor       [wrong — missed it]

def confusion_counts(y_true, y_pred):
    TP = sum(1 for p, a in zip(y_pred, y_true) if p == 1 and a == 1)
    TN = sum(1 for p, a in zip(y_pred, y_true) if p == 0 and a == 0)
    FP = sum(1 for p, a in zip(y_pred, y_true) if p == 1 and a == 0)
    FN = sum(1 for p, a in zip(y_pred, y_true) if p == 0 and a == 1)
    return TP, TN, FP, FN

def precision(TP, FP):
    # of everything predicted as versicolor, how many actually were?
    # high precision = few false alarms
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def recall(TP, FN):
    # of everything that was actually versicolor, how many did we catch?
    # high recall = few misses
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def f1_score(p, r):
    # harmonic mean of precision and recall
    # cant game it — jacking recall up (predict everything positive) destroys precision, F1 drops
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def accuracy(TP, TN, total):
    return (TP + TN) / total


def print_confusion_matrix(TP, TN, FP, FN, class_names):
    c0, c1 = class_names[0], class_names[1]
    G = "\033[92m"; R = "\033[91m"; D = "\033[2m"; B = "\033[1m"; C = "\033[96m"; X = "\033[0m"

    print(f"\n{B}{C}  -- CONFUSION MATRIX ------------------------------------{X}\n")
    print(f"  {D}{'':20}   Predicted{X}")
    print(f"  {D}{'':20}   {c0:<14}  {c1}{X}")
    print(f"  {D}{'-'*50}{X}")
    print(f"  Actual {c0:<13}   {G}{B}{TN:^10}{X}   {R}{FP:^10}{X}   {D}<- TN | FP{X}")
    print(f"  Actual {c1:<13}   {R}{FN:^10}{X}   {G}{B}{TP:^10}{X}   {D}<- FN | TP{X}")
    print(f"  {D}{'-'*50}{X}\n")
    print(f"  {G}green{X} = correct   {R}red{X} = wrong\n")


def print_metrics(TP, TN, FP, FN, p, r, f1, acc):
    G = "\033[92m"; D = "\033[2m"; B = "\033[1m"; C = "\033[96m"; X = "\033[0m"

    def bar(v, w=22):
        return f"{G}{'|' * int(v * w)}{D}{'.' * (w - int(v * w))}{X}"

    print(f"{B}{C}  -- METRICS ---------------------------------------------{X}\n")
    rows = [
        ("Accuracy",  acc, "correct predictions / total — simple but misleads on imbalanced data"),
        ("Precision", p,   "of predicted versicolor: how many were actually versicolor?"),
        ("Recall",    r,   "of all actual versicolor: how many did we catch?"),
        ("F1",        f1,  "harmonic mean of precision and recall — best single-number summary"),
    ]
    for name, val, note in rows:
        print(f"  {name:<12}  [{bar(val)}]  {B}{val:.3f}{X}  {D}<- {note}{X}")
    print(f"\n  {D}TP={TP}  TN={TN}  FP={FP}  FN={FN}{X}\n")


def print_explanations():
    B = "\033[1m"; D = "\033[2m"; C = "\033[96m"; X = "\033[0m"
    print(f"\n{B}{C}  -- METRIC EXPLANATIONS ---------------------------------{X}\n")
    items = [
        ("Precision", "When the model says versicolor, is it right?",
         "High = few false alarms. Low = model says versicolor too aggressively."),
        ("Recall",    "Of all actual versicolor flowers, how many did we find?",
         "High = nothing gets missed. Low = model is too cautious."),
        ("F1",        "Combines precision and recall into one number.",
         "If you sacrifice one to boost the other, F1 drops. Forces a real balance."),
        ("Accuracy",  "What fraction of all predictions were right?",
         "Useful on balanced data. On skewed data (99% one class), can be misleading."),
    ]
    for name, q, note in items:
        print(f"  {B}{name}{X}")
        print(f"    Question  : {q}")
        print(f"    Takeaway  : {D}{note}{X}\n")


def main():
    parser = argparse.ArgumentParser(description="Confusion matrix, precision, recall, F1 — all from scratch")
    parser.add_argument("--verbose", action="store_true", help="Also explain what each metric means")
    args = parser.parse_args()

    preds, y_test, class_names = _get_predictions()
    TP, TN, FP, FN = confusion_counts(y_test, preds)
    p   = precision(TP, FP)
    r   = recall(TP, FN)
    f1  = f1_score(p, r)
    acc = accuracy(TP, TN, len(y_test))

    print_confusion_matrix(TP, TN, FP, FN, class_names)
    print_metrics(TP, TN, FP, FN, p, r, f1, acc)
    if args.verbose:
        print_explanations()


if __name__ == "__main__":
    main()

# imported by: cli.py
# imports:     data_pipeline.py, optimised_perceptron.py
#
# python evaluate.py
# python evaluate.py --verbose