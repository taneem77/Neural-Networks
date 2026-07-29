#basically the prev file was laying interface ki this this elements u can add but this file is controlling how and when the elements are added in real time
import argparse #command line argument parser: to validate define and parse command line argument so without it wed need to manually inspect what the user has written
import sys #this is used in sys.exit(0) cus it terminated the process if there is a keyboard interrupt
import day5 #gtting the file
import ui


def parse_args(): #this is basically to understand what options the user passed when launching the program
    parser = argparse.ArgumentParser( #basically a function ki this is what the parser is understanding and accpeting
        description="Single-Layer Perceptron Classifier  |  Iris Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lr",       type=float, default=0.1,  help="Learning rate (default: 0.1)") # so in the command if they tepy that then learning rate is what u need to infer from it 
    parser.add_argument("--epochs",   type=int,   default=100,  help="Max epochs (default: 100)")
    parser.add_argument("--decay",    type=float, default=0.99, help="LR decay (default: 0.99)")
    parser.add_argument("--patience", type=int,   default=3,    help="Early stop patience (default: 3)")
    parser.add_argument("--compare",  action="store_true",      help="Compare all 3 versions") #slightly diff cus it uses a boolean flag with true or false values after comparision
    return parser.parse_args()


def main(): #basically acc to the args passed what is my program doing
    args = parse_args() #all the things we defined above thats the cli 

    if args.compare: #build the dataset
        X_train, X_test, y_train, y_test, feat_names, class_names = day5.build_pipeline(verbose=False)
        day5.compare_all(X_train, X_test, y_train, y_test) #comparing the three algorithms
        return

    # Full UI Dashboard Run
    ui.print_header()
    ui.section("LOADING DATA")
    X_train, X_test, y_train, y_test, feat_names, class_names = day5.build_pipeline(verbose=False)
    ui.print_data_summary(X_train, X_test, y_train, y_test, feat_names, class_names)
    ui.print_model_config(args.lr, args.epochs, args.decay, args.patience) #example of a connection between ui and cli

    model = day5.ConvergentPerceptron( # this is basically user gets command and configures the model without actually editting the source code
        lr=args.lr, epochs=args.epochs,
        decay=args.decay, patience=args.patience
    )
    ui.train_with_bar(model, X_train, y_train)

    train_acc = model.accuracy(X_train, y_train)# how well does the trained model train the samples
    test_acc  = model.accuracy(X_test,  y_test) #how well does trained model claisfy unseens test samples
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
