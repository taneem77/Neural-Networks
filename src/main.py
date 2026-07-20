import argparse
import sys
import day5
import ui


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-Layer Perceptron Classifier  |  Iris Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lr",       type=float, default=0.1,  help="Learning rate (default: 0.1)")
    parser.add_argument("--epochs",   type=int,   default=100,  help="Max epochs (default: 100)")
    parser.add_argument("--decay",    type=float, default=0.99, help="LR decay (default: 0.99)")
    parser.add_argument("--patience", type=int,   default=3,    help="Early stop patience (default: 3)")
    parser.add_argument("--compare",  action="store_true",      help="Compare all 3 versions")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.compare:
        X_train, X_test, y_train, y_test, feat_names, class_names = day5.build_pipeline(verbose=False)
        day5.compare_all(X_train, X_test, y_train, y_test)
        return

    # Full UI Dashboard Run
    ui.print_header()
    ui.section("LOADING DATA")
    X_train, X_test, y_train, y_test, feat_names, class_names = day5.build_pipeline(verbose=False)
    ui.print_data_summary(X_train, X_test, y_train, y_test, feat_names, class_names)
    ui.print_model_config(args.lr, args.epochs, args.decay, args.patience)

    model = day5.ConvergentPerceptron(
        lr=args.lr, epochs=args.epochs,
        decay=args.decay, patience=args.patience
    )
    ui.train_with_bar(model, X_train, y_train)

    train_acc = model.accuracy(X_train, y_train)
    test_acc  = model.accuracy(X_test,  y_test)
    ui.print_accuracy(train_acc, test_acc)
    ui.print_learning_curve(model)
    ui.print_predictions(model, X_test, y_test, class_names)
    ui.menu(model, X_train, X_test, y_train, y_test, feat_names, class_names, train_acc, test_acc)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Goodbye!\n")
        sys.exit(0)