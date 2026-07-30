#gives  iris measurement and get actual predction of setosa vs versicolor
#see how this is diff is cus earlier we build perceptron and then improve training and then add iris data and then normalise and track convergence 
#in this we take a trained model accept the unseen flower and then scale the measurements , calculate z classify and show the decision
#intuitively its now i have weights how do i use them to clasify a new flower

import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler #scale user inputs
from data_pipeline import build_pipeline #irir data
from optimised_perceptron import OptimisedPerceptron

FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"] # 4 features
CLASSES  = ["setosa", "versicolor"] #0 or 1 with the type of flower


def get_trained_model(): #prepares model before predcition
    import io, sys
    X_train, _, y_train, _, _, _ = build_pipeline(verbose=False)
    model = OptimisedPerceptron(lr=0.1, epochs=100)
    buf = io.StringIO() #prints without clutter
    sys.stdout = buf
    model.fit(X_train, y_train)
    sys.stdout = sys.__stdout__
    scaler = StandardScaler() #creates a scaler of new input which is later used my user 
    scaler.fit(X_train)
    return model, scaler


def predict_one(model, scaler, values): #prediction logic 
    x_scaled = scaler.transform(np.array(values).reshape(1, -1)) #standardise output 
    z = np.dot(model.weights, x_scaled[0]) + model.bias #converts values into numpy array values
    return (1 if z >= 0 else 0), z #rem z decides which side of decision boundary the flower lies on
    #z<0 is setosa


def show_result(values, pred, z): #this is to make output look nice
    G = "\033[92m"; R = "\033[91m"; C = "\033[96m"; B = "\033[1m"; D = "\033[2m"; X = "\033[0m"
    print(f"\n{B}{C}  -- PREDICTION ------------------------------------------{X}\n")
    for name, val in zip(FEATURES, values):
        print(f"  {name:<26}  {val}")
    color = G if pred == 1 else C
    confidence = "very confident" if abs(z) > 2 else "confident" if abs(z) > 0.8 else "borderline" #acc to magnitude of z we have confidence and for confidence we use absolute value
    #this is intuitive (confidence)
    print(f"\n  Class     ->  {color}{B}{CLASSES[pred].upper()}{X}")
    print(f"  z-score   ->  {z:+.4f}  {D}(distance from decision boundary){X}")
    print(f"  Margin    ->  {confidence}  {D}|z| = {abs(z):.2f}{X}")
    width = 28
    center = width // 2
    pos = int(center + max(min(z, 4.0), -4.0) / 4.0 * center) #this is clamping the z value between any two biundaries so it calculates more accurately 
    bar = list("." * width)
    bar[center] = "|"
    bar[pos] = "O"
    dot_color = G if z >= 0 else R
    print(f"\n  {D}setosa <-{X}  {dot_color}[{''.join(bar)}]{X}  {D}-> versicolor{X}")
    print(f"  {D}left of | = setosa  right of | = versicolor{X}\n")


def interactive_mode(model, scaler): #interactive terminal mode
    Y = "\033[93m"; C = "\033[96m"; B = "\033[1m"; D = "\033[2m"; X = "\033[0m"
    print(f"\n{B}{C}  Interactive Mode  {D}(q to quit){X}")
    print(f"  Enter: sepal_len  sepal_wid  petal_len  petal_wid\n")
    while True:
        raw = input("  -> ").strip()
        if raw.lower() == "q":
            print(f"\n  {Y}Done.{X}\n")
            break
        try:
            values = [float(v) for v in raw.split()]
            if len(values) != 4:
                print(f"  Need 4 values, got {len(values)}. Try again.\n")
                continue
            pred, z = predict_one(model, scaler, values)
            show_result(values, pred, z)
        except ValueError:
            print(f"  Couldn't parse those — enter 4 numbers separated by spaces.\n")


def main():
    parser = argparse.ArgumentParser(description="Predict Iris class from 4 measurements")
    parser.add_argument("--features", type=float, nargs=4, metavar=("SL", "SW", "PL", "PW"))
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    print(f"  \033[2mLoading...\033[0m", end="\r", flush=True)
    model, scaler = get_trained_model()
    print(f"  \033[92m  Ready\033[0m        ")

    if args.interactive:
        interactive_mode(model, scaler)
    elif args.features:
        pred, z = predict_one(model, scaler, args.features)
        show_result(args.features, pred, z)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

# imported by: cli.py
# imports:     data_pipeline.py, optimised_perceptron.py
