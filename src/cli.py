import argparse
import sys

# single entry point for the whole project
# zero logic here — imports from every module and connects them through subcommands
# same pattern as git: one tool, many subcommands
#   cli.py train / cli.py predict / cli.py eval / cli.py history / cli.py info


def cmd_train(args): #runs training flow
    #this is what loads iris data set, creates optimised percetrpon , trains model, displays testing accuracy etc 
    from data_pipeline import build_pipeline
    from optimised_perceptron import OptimisedPerceptron
    from terminal_ui import (
        print_header, print_data_summary, print_model_config,
        train_with_bar, print_accuracy, print_learning_curve,
        print_predictions, menu, section
    )
    print_header()
    section("LOADING DATA")
    X_train, X_test, y_train, y_test, feat_names, class_names = build_pipeline(verbose=False)
    print_data_summary(X_train, X_test, y_train, y_test, feat_names, class_names)
    print_model_config(args.lr, args.epochs)
    model = OptimisedPerceptron(lr=args.lr, epochs=args.epochs)
    train_with_bar(model, X_train, y_train)
    train_acc = model.accuracy(X_train, y_train)
    test_acc  = model.accuracy(X_test, y_test)
    print_accuracy(train_acc, test_acc)
    print_learning_curve(model)
    print_predictions(model, X_test, y_test, class_names)
    if args.save:
        from persist import save_model
        save_model(model, X_train)
    menu(model, X_train, X_test, y_train, y_test, feat_names, class_names, train_acc, test_acc)

#loads the saved model and perfroms prediticon , as in predict using four features or interactive mode 
def cmd_predict(args):
    from predict import get_trained_model, predict_one, show_result, interactive_mode
    print(f"  \033[2mLoading...\033[0m", end="\r", flush=True)
    model, scaler = get_trained_model()
    print(f"  \033[92m  Ready\033[0m        ")
    if args.interactive:
        interactive_mode(model, scaler)
    elif args.features:
        pred, z = predict_one(model, scaler, args.features)
        show_result(args.features, pred, z)
    else:
        print("  Provide --features or --interactive\n")


def cmd_eval(args): #this is wehre if --verbose is used its called, it calculates confusion matrix, precision, recall 
    from evaluate import (
        _get_predictions, confusion_counts, precision, recall,
        f1_score, accuracy, print_confusion_matrix, print_metrics, print_explanations
    )
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


def cmd_history(args):
    from logger import show_history, train_and_log
    if args.log:
        train_and_log(args.lr, args.epochs)
    else:
        show_history()


def cmd_info(args):
    from persist import show_info
    show_info(args.path)


def main():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="\033[1m\033[96m  Perceptron Classifier -- Iris Dataset\033[0m",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  train      train model, open interactive terminal UI
  predict    classify a new flower from measurements
  eval       confusion matrix + precision, recall, F1
  history    view past training logs (or record a new run)
  info       show saved model file metadata

examples:
  python cli.py train
  python cli.py train --lr 0.05 --save
  python cli.py predict --features 5.1 3.5 1.4 0.2
  python cli.py predict --interactive
  python cli.py eval --verbose
  python cli.py history
  python cli.py history --log --lr 0.01
  python cli.py info
        """
    )

    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train")
    p_train.add_argument("--lr",     type=float, default=0.1)
    p_train.add_argument("--epochs", type=int,   default=100)
    p_train.add_argument("--save",   action="store_true", help="Save weights after training")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--features",    type=float, nargs=4, metavar=("SL","SW","PL","PW"))
    p_pred.add_argument("--interactive", action="store_true")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--verbose", action="store_true")

    p_hist = sub.add_parser("history")
    p_hist.add_argument("--log",     action="store_true")
    p_hist.add_argument("--lr",      type=float, default=0.1)
    p_hist.add_argument("--epochs",  type=int,   default=100)

    p_info = sub.add_parser("info")
    p_info.add_argument("--path", type=str, default="saved_model.npz")

    args = parser.parse_args()

    dispatch = {
        "train":   cmd_train,
        "predict": cmd_predict,
        "eval":    cmd_eval,
        "history": cmd_history,
        "info":    cmd_info,
    }

    if args.command in dispatch:
        try:
            dispatch[args.command](args)
        except KeyboardInterrupt:
            print(f"\n\n  \033[93mInterrupted. Bye!\033[0m\n")
            sys.exit(0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

# full import map:
#   cli.py
#    train   -> terminal_ui.py, optimised_perceptron.py, data_pipeline.py, persist.py
#    predict -> predict.py  -> data_pipeline.py, optimised_perceptron.py
#    eval    -> evaluate.py -> data_pipeline.py, optimised_perceptron.py
#    history -> logger.py   -> data_pipeline.py, optimised_perceptron.py
#    info    -> persist.py
