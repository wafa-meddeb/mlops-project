"""Point d'entree CLI : python main.py --step {prepare,train,evaluate,all}"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    log_experiment,
    prepare_data,
    save_model,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn")
    parser.add_argument(
        "--step",
        choices=["prepare", "train", "evaluate", "all"],
        default="all",
        help="Etape a executer",
    )
    parser.add_argument("--data", default="Churn_Modelling.csv")
    parser.add_argument("--model", default="classifier.joblib")
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = prepare_data(args.data)

    if args.step == "prepare":
        print(f"Train : {X_train.shape} | Test : {X_test.shape}")

    elif args.step == "train":
        model = train_model(X_train, y_train, args.n_estimators)
        save_model(model, args.model)
        log_experiment(model, X_test, y_test)

    elif args.step == "evaluate":
        model = load_model(args.model)
        evaluate_model(model, X_test, y_test)

    else:  # all
        model = train_model(X_train, y_train, args.n_estimators)
        evaluate_model(model, X_test, y_test)
        save_model(model, args.model)
        log_experiment(model, X_test, y_test)


if __name__ == "__main__":
    main()
