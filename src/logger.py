import numpy as np
import argparse
import logging
import os
import time
from datetime import datetime
from data_pipeline import build_pipeline
from optimised_perceptron import OptimisedPerceptron

LOG_DIR = "logs"

# why logging instead of print?
# print statements disappear when the terminal closes
# logging writes to a file — every training run is permanently saved
# this is standard ML practice called "experiment tracking"
# you can run 10 experiments with different learning rates and compare them later


def _make_logger():
    # each run gets its own timestamped file: logs/run_20260720_143022.log
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"run_{stamp}.log")

    logger = logging.getLogger(f"perceptron_{stamp}")  # unique name avoids handler stacking
    logger.setLevel(logging.DEBUG)

    # file handler: gets everything including per-epoch DEBUG lines
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%Y-%m-%d %H:%M:%S"))

    # stream handler: only INFO and above — per-epoch detail stays in file, terminal stays clean
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("  %(levelname)-8s  %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, log_path


def train_and_log(lr=0.1, epochs=100):
    logger, log_path = _make_logger()
    logger.info("=== NEW RUN ===")
    logger.info(f"lr={lr}  epochs={epochs}")

    X_train, X_test, y_train, y_test, _, _ = build_pipeline(verbose=False)
    logger.info(f"Data: {X_train.shape[0]} train  {X_test.shape[0]} test  {X_train.shape[1]} features")

    model = OptimisedPerceptron(lr=lr, epochs=epochs)

    t_start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t_start

    # log per-epoch error counts to file (DEBUG level — won't show in terminal)
    for i, errors in enumerate(model.errors_per_epoch):
        logger.debug(f"epoch {i+1:>3}  errors={errors}")

    test_acc  = model.accuracy(X_test, y_test)
    train_acc = model.accuracy(X_train, y_train)

    logger.info(f"Done in {elapsed:.3f}s  |  epochs run={len(model.errors_per_epoch)}")
    logger.info(f"Train acc={train_acc:.1f}%  |  Test acc={test_acc:.1f}%")
    logger.info(f"Weights: {np.round(model.weights, 4).tolist()}")
    logger.info(f"Bias: {model.bias:.4f}")

    print(f"\n  \033[2mLog -> {os.path.abspath(log_path)}\033[0m\n")
    return log_path


def show_history():
    R = "\033[91m"; C = "\033[96m"; D = "\033[2m"; B = "\033[1m"; X = "\033[0m"

    if not os.path.exists(LOG_DIR) or not os.listdir(LOG_DIR):
        print(f"  {R}No logs in ./{LOG_DIR}/ — run without --history first.{X}\n")
        return

    files = sorted(f for f in os.listdir(LOG_DIR) if f.endswith(".log"))
    print(f"\n{B}{C}  -- TRAINING HISTORY ({len(files)} run{'s' if len(files)!=1 else ''}) --------------------------{X}\n")

    for fname in files:
        path     = os.path.join(LOG_DIR, fname)
        saved_at = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
        summary  = []
        with open(path) as f:
            for line in f:
                for kw in ["lr=", "Done in", "Train acc", "Weights"]:
                    if kw in line and "INFO" in line:
                        summary.append(line.strip().split("  ")[-1])
        print(f"  {C}{fname}{X}  {D}({saved_at}){X}")
        for line in summary:
            print(f"    {D}{line}{X}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Train with file logging and run history")
    parser.add_argument("--lr",      type=float, default=0.1)
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--history", action="store_true", help="Show all past runs")
    args = parser.parse_args()

    if args.history:
        show_history()
    else:
        train_and_log(args.lr, args.epochs)


if __name__ == "__main__":
    main()

# imported by: cli.py
# imports:     data_pipeline.py, optimised_perceptron.py