"""Pipeline ML modulaire - Churn Modelling (Atelier 2)."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
