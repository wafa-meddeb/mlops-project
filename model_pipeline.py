"""Pipeline ML modulaire - Churn Modelling (Atelier 2)."""

import time
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

DROP_COLS = ["RowNumber", "CustomerId", "Surname", "Geography"]
TARGET = "Exited"


def prepare_data(csv_path="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """Charge le CSV, nettoie et retourne X_train, X_test, y_train, y_test."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df = df.dropna()

    # Encodage Gender : Female=0, Male=1
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un RandomForestClassifier et le retourne."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle, affiche les métriques et retourne l'accuracy."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {acc * 100:.2f} %")
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return acc


def save_model(model, path="classifier.joblib"):
    """Sauvegarde le modèle avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")


def load_model(path="classifier.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    return joblib.load(path)


FEATURES = [
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


def predict(model, features):
    """Predit le depart d'un client : retourne (classe 0/1, probabilite de depart).

    `features` est une liste de 9 valeurs, dans l'ordre de FEATURES.
    """
    X = pd.DataFrame([features], columns=FEATURES)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    return prediction, probability


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "churn_prediction"


def setup_mlflow(tracking_uri=MLFLOW_TRACKING_URI):
    """Configure MLflow : backend SQLite local et experience du projet."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


ELASTIC_URL = "http://localhost:9200"
ELASTIC_INDEX = "mlflow-metrics"


def send_to_elasticsearch(doc, url=ELASTIC_URL, index=ELASTIC_INDEX):
    """Envoie un document dans Elasticsearch (sans bloquer le pipeline si KO)."""
    try:
        from elasticsearch import Elasticsearch

        Elasticsearch(url).index(index=index, document=doc)
        print(f"Run envoye a Elasticsearch (index {index})")
    except Exception as exc:
        print(f"Elasticsearch indisponible, envoi ignore : {exc}")


def log_experiment(model, X_test, y_test, run_name="random_forest"):
    """Enregistre params, metriques et modele dans MLflow puis Elasticsearch."""
    import mlflow
    import mlflow.sklearn

    setup_mlflow()
    y_pred = model.predict(X_test)
    params = {
        "n_estimators": int(model.n_estimators),
        "random_state": int(model.random_state),
    }
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }
    start = time.time()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

    doc = {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_name": run_name,
        "experiment_id": experiment_id,
        "experiment_name": EXPERIMENT_NAME,
        "status": "FINISHED",
        "duration_seconds": round(time.time() - start, 3),
        "params": params,
        "metrics": metrics,
    }
    doc.update({f"param_{k}": str(v) for k, v in params.items()})
    doc.update({f"metric_{k}": v for k, v in metrics.items()})
    send_to_elasticsearch(doc)
