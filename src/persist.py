#train → save what the model learned → close the program → reload it later → use it without retraining.

import numpy as np
import argparse #--save, --load, --info
import os
import time
from sklearn.preprocessing import StandardScaler # feature normalization
from data_pipeline import build_pipeline #get the files
from optimised_perceptron import OptimisedPerceptron
from export_kernel import export_to_kernel   # NEW

MODEL_PATH = "saved_model.npz" #basically whatever training was done is saved into this file


def _train_silently(): #trains perceptron without printing training output
    import io, sys
    X_train, X_test, y_train, y_test, _, _ = build_pipeline(verbose=False)
    model = OptimisedPerceptron(lr=0.1, epochs=100)
    buf = io.StringIO() #this is like a print statement in memory buffer
    sys.stdout = buf
    model.fit(X_train, y_train)
    sys.stdout = sys.__stdout__
    return model, X_train, X_test, y_train, y_test


def save_model(model, X_train, path=MODEL_PATH):
    # save weights and bias alongside the scaler's mean and std
    # reason: during training we called scaler.fit_transform(X_train), which found the mean and std
    # if we only save weights and reload them later, any new data we pass would be raw (unscaled)
    # and the dot product would give wrong answers silently — no error thrown
    # so we save scaler params too, to apply the exact same scaling at prediction time
    scaler = StandardScaler()
    scaler.fit(X_train)

    np.savez( #creats a .npz file using numpy arrays
        path,
        weights          = model.weights,
        bias             = np.array([model.bias]),
        lr               = np.array([model.lr]),
        epochs_run       = np.array([len(model.errors_per_epoch)]),
        scaler_mean      = scaler.mean_,
        scaler_std       = scaler.scale_,
        errors_per_epoch = np.array(model.errors_per_epoch),
    )

    # NEW : automatically create the kernel header from learned weights
    export_to_kernel(path, "kernel/exported_weights.h")

    G = "\033[92m"; C = "\033[96m"; D = "\033[2m"; X = "\033[0m" #coloured output with ansi terminal codes
    print(f"\n  {G}  Saved{X}")
    print(f"  Path   ->  {C}{os.path.abspath(path)}{X}")
    print(f"  Size   ->  {os.path.getsize(path)/1024:.2f} KB")
    print(f"  Format ->  .npz  {D}(NumPy compressed archive — stores multiple arrays in one file){X}\n")


def load_model(path=MODEL_PATH): #loads an exisiting model so it checks if there is a file already saved w training
    R = "\033[91m"; G = "\033[92m"; C = "\033[96m"; D = "\033[2m"; X = "\033[0m"

    if not os.path.exists(path):
        print(f"  {R}No saved model at {path} — run --save first.{X}\n")
        return None, None

    data  = np.load(path) #the following lines are to put prev learned values back into an empty perceptron created so now new model -> loads old wieghts -> ready to predict
    model = OptimisedPerceptron(lr=float(data["lr"][0]), epochs=100)
    model.weights          = data["weights"]
    model.bias             = float(data["bias"][0])
    model.errors_per_epoch = list(data["errors_per_epoch"])

    # rebuild scaler from saved params so new data gets scaled identically to training time
    scaler                = StandardScaler()
    scaler.mean_          = data["scaler_mean"]
    scaler.scale_         = data["scaler_std"]
    scaler.var_           = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    saved_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
    print(f"\n  {G}  Loaded  {C}{os.path.abspath(path)}{X}")
    print(f"  Saved at   ->  {saved_at}")
    print(f"  Epochs run ->  {int(data['epochs_run'][0])}")
    print(f"  Weights    ->  {np.round(model.weights, 4)}\n")
    return model, scaler


def show_info(path=MODEL_PATH): #this is just to print what is stored
    R = "\033[91m"; C = "\033[96m"; B = "\033[1m"; D = "\033[2m"; X = "\033[0m"

    if not os.path.exists(path):
        print(f"  {R}No saved model at {path}{X}\n")
        return

    data     = np.load(path)
    saved_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))) #metadata of file

    print(f"\n{B}{C}   SAVED MODEL INFO {X}\n")
    print(f"  File       ->  {os.path.abspath(path)}")
    print(f"  Size       ->  {os.path.getsize(path)/1024:.2f} KB")
    print(f"  Saved at   ->  {saved_at}")
    print(f"  Arrays     ->  {list(data.files)}")
    print(f"  Weights    ->  {np.round(data['weights'], 4)}")
    print(f"  Bias       ->  {float(data['bias'][0]):.4f}")
    print(f"  LR used    ->  {float(data['lr'][0])}")
    print(f"  Epochs run ->  {int(data['epochs_run'][0])}\n")


def main():
    parser = argparse.ArgumentParser(description="Save and load trained perceptron weights")
    parser.add_argument("--save", action="store_true", help="Train and save to disk") # ex: build iris dataset, train on optimised perceptrons, obtain the bias and weight and save model info then retrieve it to show test accuracy
    parser.add_argument("--load", action="store_true", help="Load saved model and verify accuracy") #show accuracy of freshly trained model and saved model
    parser.add_argument("--info", action="store_true", help="Show saved file metadata") #Inspection
    parser.add_argument("--path", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    if args.save:
        print(f"  \033[2mTraining...\033[0m", end="\r", flush=True)
        model, X_train, X_test, y_train, y_test = _train_silently()
        print(f"  \033[92m  Training complete\033[0m            ")
        save_model(model, X_train, args.path)
        print(f"  Test accuracy  ->  \033[92m{model.accuracy(X_test, y_test):.1f}%\033[0m\n")

    elif args.load:
        model, scaler = load_model(args.path)
        if model:
            _, X_test, _, y_test, _, _ = build_pipeline(verbose=False)
            print(f"  Test accuracy (loaded)  ->  \033[92m{model.accuracy(X_test, y_test):.1f}%\033[0m")
            print(f"  \033[2m(identical to freshly trained — weights are stored exactly)\033[0m\n")

    elif args.info:
        show_info(args.path)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

# imported by: cli.py
# imports:     data_pipeline.py, optimised_perceptron.py