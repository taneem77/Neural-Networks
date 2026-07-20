

import numpy as np
import sys
import os
import time


from data_pipeline import build_pipeline
from train import ConvergentPerceptron


# Format: \033[<code>m
#   0  = reset all
#   1  = bold
#   2  = dim
#   9x = bright colors
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    WHITE   = "\033[97m"


# ─── LAYOUT HELPERS ───────────────────────────────────────────────────────────
def terminal_width():
    """Get terminal width, fall back to 80 if not in a real terminal."""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80

def rule(char="─", color=C.DIM):
    """Print a full-width horizontal rule."""
    print(f"{color}{char * terminal_width()}{C.RESET}")

def section(title):
    """Print a section header with a rule below."""
    print(f"\n{C.BOLD}{C.YELLOW}▸ {title}{C.RESET}")
    rule("·", C.DIM)

def clear():
    os.system("clear")

def status(msg, kind="info"):
    """Print a status line with an icon."""
    icon = {"info": f"{C.CYAN}◈", "ok": f"{C.GREEN}✔",
            "warn": f"{C.YELLOW}⚠", "err": f"{C.RED}✘"}.get(kind, "·")
    print(f"  {icon}  {msg}{C.RESET}")


# ─── PROGRESS BAR ─────────────────────────────────────────────────────────────
def progress_bar(current, total, width=38, label="Training"):
    """
    Animated progress bar using \r (carriage return).
    \r moves the cursor to the start of the current line without a newline.
    Next print overwrites it → looks like animation.
    """
    filled = int(width * current / total)
    bar    = f"{C.GREEN}{'█' * filled}{C.DIM}{'░' * (width - filled)}{C.RESET}"
    pct    = int(100 * current / total)
    print(f"\r  {C.BOLD}{label}{C.RESET}  [{bar}]  {C.WHITE}{pct:3d}%{C.RESET}  "
          f"epoch {current}/{total}", end="", flush=True)


# ─── HEADER ───────────────────────────────────────────────────────────────────
def print_header():
    clear()
    rule("═", C.CYAN)
    print(f"{C.BOLD}{C.CYAN}  ⬡  PERCEPTRON CLASSIFIER  "
          f"⬡  Iris Dataset  ⬡  Single Layer{C.RESET}")
    print(f"{C.DIM}  Iris: Setosa vs Versicolor  "
          f"|  4 features  |  80/20 split  |  ConvergentPerceptron{C.RESET}")
    rule("═", C.CYAN)


# ─── DATA SUMMARY ─────────────────────────────────────────────────────────────
def print_data_summary(X_train, X_test, y_train, y_test, feat_names, class_names):
    section("DATASET  —  Iris (Binary)")
    rows = [
        ("Features",          ", ".join(feat_names)),
        ("Classes",           f"{class_names[0]}  vs  {class_names[1]}"),
        ("Total samples",     f"{len(X_train) + len(X_test)}"),
        ("Training samples",  f"{C.GREEN}{len(X_train)}{C.RESET}  (80%)"),
        ("Test samples",      f"{C.CYAN}{len(X_test)}{C.RESET}  (20%)"),
        ("Normalisation",     "StandardScaler  (mean=0, std=1)"),
    ]
    for label, value in rows:
        print(f"  {label:<22}  {value}")


# ─── MODEL CONFIG ─────────────────────────────────────────────────────────────
def print_model_config(lr, epochs, decay, patience):
    section("MODEL CONFIG  —  ConvergentPerceptron")
    rows = [
        ("Learning rate",    str(lr)),
        ("Max epochs",       str(epochs)),
        ("LR decay",         f"{decay}  (lr shrinks {(1-decay)*100:.0f}% per epoch)"),
        ("Early stop",       f"patience={patience}  ({patience} perfect epochs → stop)"),
        ("Activation",       "Step function  (z ≥ 0 → 1, else 0)"),
        ("Update rule",      "w += lr × error × confidence_scale × x"),
    ]
    for label, value in rows:
        print(f"  {label:<22}  {value}")


# ─── TRAINING WITH ANIMATED BAR ───────────────────────────────────────────────
def train_with_bar(model, X_train, y_train):
    """
    Train ConvergentPerceptron but show a live progress bar.
    We do this by monkey-patching: we wrap the fit loop externally.
    Simpler approach: just show the bar during fit and print result after.
    """
    section("TRAINING")

    # We'll run fit() but intercept each epoch using a subclass trick.
    # Simpler version: just show a fake timed bar, then actually train.
    # Real version for later: override fit() to yield after each epoch.

    # For now: show bar at 0%, run training, then jump to 100%.
    progress_bar(0, model.epochs)
    time.sleep(0.1)

    # Redirect stdout temporarily so fit()'s own prints don't break the bar
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    model.fit(X_train, y_train)
    fit_output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # Show completed bar
    epochs_used = len(model.history['errors'])
    progress_bar(epochs_used, model.epochs)
    print()   # newline after bar

    # Now print what fit() said (converged message etc.)
    for line in fit_output.strip().split('\n'):
        if line.strip():
            status(line.strip(), "ok")


# ─── RESULTS ──────────────────────────────────────────────────────────────────
def print_accuracy(train_acc, test_acc):
    section("RESULTS")

    def color(a):
        return C.GREEN if a >= 95 else (C.YELLOW if a >= 80 else C.RED)

    def bar(a, width=28):
        filled = int(width * a / 100)
        c = color(a)
        return f"{c}{'█' * filled}{C.DIM}{'░' * (width - filled)}{C.RESET}"

    print(f"\n  {'Train Accuracy':<16}  [{bar(train_acc)}]  "
          f"{color(train_acc)}{C.BOLD}{train_acc:.1f}%{C.RESET}")
    print(f"  {'Test Accuracy':<16}  [{bar(test_acc)}]  "
          f"{color(test_acc)}{C.BOLD}{test_acc:.1f}%{C.RESET}")


def print_weights(model, feat_names):
    section("LEARNED WEIGHTS")
    print(f"  {'Bias':<30}  {model.bias:+.4f}\n")
    for name, w in zip(feat_names, model.weights):
        bar_len = int(min(abs(w) * 10, 20))
        bar = (f"{C.GREEN}{'▶' * bar_len}{C.RESET}" if w > 0
               else f"{C.RED}{'◀' * bar_len}{C.RESET}")
        print(f"  {name:<30}  {w:+.4f}  {bar}")


def print_learning_curve(model):
    """
    ASCII bar chart of errors per epoch.
    Uses model.history['errors'] from Day 4's ConvergentPerceptron.
    """
    section("LEARNING CURVE  (misclassifications per epoch)")
    errors  = model.history['errors']
    max_e   = max(errors) if max(errors) > 0 else 1
    n_cols  = min(len(errors), 52)
    step    = max(1, len(errors) // n_cols)
    sampled = errors[::step][:n_cols]
    rows    = 7

    for row in range(rows, -1, -1):
        threshold = max_e * row / rows
        line = "  "
        for e in sampled:
            line += (f"{C.GREEN}█{C.RESET}" if e > threshold
                     else f"{C.DIM}·{C.RESET}")
        print(line)
    print(f"  {C.DIM}epoch 1{' ' * (n_cols - 15)}epoch {len(errors)}{C.RESET}")


def print_lr_curve(model):
    """Show how learning rate decayed over training."""
    section("LEARNING RATE DECAY")
    lrs     = model.history['lr']
    n_cols  = min(len(lrs), 52)
    step    = max(1, len(lrs) // n_cols)
    sampled = lrs[::step][:n_cols]
    max_lr  = max(lrs)
    rows    = 5

    for row in range(rows, -1, -1):
        threshold = max_lr * row / rows
        line = "  "
        for lr in sampled:
            line += (f"{C.CYAN}█{C.RESET}" if lr > threshold
                     else f"{C.DIM}·{C.RESET}")
        print(line)
    print(f"  {C.DIM}start: {max_lr:.4f}{' ' * (n_cols - 20)}end: {lrs[-1]:.4f}{C.RESET}")


def print_predictions(model, X_test, y_test, class_names, n=12):
    section(f"SAMPLE PREDICTIONS  (first {n} test samples)")
    preds = model.predict(X_test)
    print(f"  {C.DIM}{'#':<4}  {'Actual':<12}  {'Predicted':<12}  Result{C.RESET}")
    rule("·", C.DIM)
    for i in range(min(n, len(X_test))):
        actual    = class_names[y_test[i]]
        predicted = class_names[preds[i]]
        match     = preds[i] == y_test[i]
        verdict   = f"{C.GREEN}✔ correct{C.RESET}" if match else f"{C.RED}✘ wrong{C.RESET}"
        print(f"  {i+1:<4}  {actual:<12}  {predicted:<12}  {verdict}")


# ─── RETRAIN ──────────────────────────────────────────────────────────────────
def retrain_prompt(X_train, X_test, y_train, y_test, feat_names, class_names):
    """Ask user for new hyperparams, retrain, return new model + accuracies."""
    section("RETRAIN")
    try:
        lr_in  = input(f"  New learning rate  [default 0.1]  : ").strip()
        ep_in  = input(f"  New max epochs     [default 100]  : ").strip()
        dc_in  = input(f"  LR decay           [default 0.99] : ").strip()
        pt_in  = input(f"  Patience           [default 3]    : ").strip()

        lr      = float(lr_in) if lr_in else 0.1
        epochs  = int(ep_in)   if ep_in else 100
        decay   = float(dc_in) if dc_in else 0.99
        patience = int(pt_in)  if pt_in else 3

        model = ConvergentPerceptron(lr=lr, epochs=epochs, decay=decay, patience=patience)
        train_with_bar(model, X_train, y_train)
        ta = model.accuracy(X_train, y_train)
        te = model.accuracy(X_test,  y_test)
        print_accuracy(ta, te)
        return model, ta, te

    except ValueError:
        status("Invalid input — keeping current model", "warn")
        return None, None, None


# ─── MENU ─────────────────────────────────────────────────────────────────────
def menu(model, X_train, X_test, y_train, y_test, feat_names, class_names,
         train_acc, test_acc):

    options = [
        ("1", "Show accuracy"),
        ("2", "Show learned weights"),
        ("3", "Show learning curve"),
        ("4", "Show LR decay curve"),
        ("5", "Show sample predictions"),
        ("6", "Retrain with new hyperparameters"),
        ("q", "Exit"),
    ]

    while True:
        print()
        rule("─", C.DIM)
        print(f"\n  {C.BOLD}What would you like to do?{C.RESET}\n")
        for key, label in options:
            color = C.RED if key == "q" else C.CYAN
            print(f"    {color}[{key}]{C.RESET}  {label}")
        print()

        choice = input(f"  {C.BOLD}→ {C.RESET}").strip().lower()

        if choice == "q":
            print(f"\n  {C.MAGENTA}See you! 👋{C.RESET}\n")
            sys.exit(0)

        elif choice == "1":
            print_accuracy(train_acc, test_acc)

        elif choice == "2":
            print_weights(model, feat_names)

        elif choice == "3":
            print_learning_curve(model)

        elif choice == "4":
            print_lr_curve(model)

        elif choice == "5":
            print_predictions(model, X_test, y_test, class_names)

        elif choice == "6":
            new_model, ta, te = retrain_prompt(
                X_train, X_test, y_train, y_test, feat_names, class_names)
            if new_model:
                model      = new_model
                train_acc  = ta
                test_acc   = te

        else:
            status("Unknown option — try again", "warn")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print_header()

    # Load data (Day 3)
    section("LOADING DATA")
    X_train, X_test, y_train, y_test, feat_names, class_names = build_pipeline(verbose=False)
    print_data_summary(X_train, X_test, y_train, y_test, feat_names, class_names)

    # Model config
    LR, EPOCHS, DECAY, PATIENCE = 0.1, 100, 0.99, 3
    print_model_config(LR, EPOCHS, DECAY, PATIENCE)

    # Train (Day 4)
    model = ConvergentPerceptron(lr=LR, epochs=EPOCHS, decay=DECAY, patience=PATIENCE)
    train_with_bar(model, X_train, y_train)

    # Auto-show results
    train_acc = model.accuracy(X_train, y_train)
    test_acc  = model.accuracy(X_test,  y_test)
    print_accuracy(train_acc, test_acc)
    print_learning_curve(model)
    print_predictions(model, X_test, y_test, class_names)

    # Interactive menu
    menu(model, X_train, X_test, y_train, y_test, feat_names, class_names,
         train_acc, test_acc)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n  {C.YELLOW}Interrupted (Ctrl+C). Goodbye!{C.RESET}\n")
        sys.exit(0)
