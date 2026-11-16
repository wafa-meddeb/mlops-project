"""Tests unitaires de model_pipeline.py."""

import numpy as np
import pandas as pd

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def make_csv(path, n=100):
    """Cree un petit CSV synthetique au meme schema que Churn_Modelling.csv."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "RowNumber": range(n),
            "CustomerId": range(n),
            "Surname": "X",
            "CreditScore": rng.integers(400, 850, n),
            "Geography": rng.choice(["France", "Spain"], n),
            "Gender": rng.choice(["Female", "Male"], n),
            "Age": rng.integers(18, 80, n),
            "Tenure": rng.integers(0, 10, n),
            "Balance": rng.random(n) * 1e5,
            "NumOfProducts": rng.integers(1, 4, n),
            "HasCrCard": rng.integers(0, 2, n),
            "IsActiveMember": rng.integers(0, 2, n),
            "EstimatedSalary": rng.random(n) * 1e5,
            "Exited": rng.integers(0, 2, n),
        }
    )
    df.to_csv(path, index=False)


def test_prepare_data(tmp_path):
    csv = tmp_path / "data.csv"
    make_csv(csv)
    X_train, X_test, y_train, y_test = prepare_data(str(csv))
    assert len(X_train) == 80 and len(X_test) == 20
    assert "Exited" not in X_train.columns
    assert X_train.shape[1] == 9


def test_train_evaluate(tmp_path):
    csv = tmp_path / "data.csv"
    make_csv(csv)
    X_train, X_test, y_train, y_test = prepare_data(str(csv))
    model = train_model(X_train, y_train, n_estimators=10)
    assert 0 <= evaluate_model(model, X_test, y_test) <= 1


def test_save_load(tmp_path):
    csv = tmp_path / "data.csv"
    make_csv(csv)
    X_train, X_test, y_train, y_test = prepare_data(str(csv))
    model = train_model(X_train, y_train, n_estimators=10)
    path = str(tmp_path / "model.joblib")
    save_model(model, path)
    loaded = load_model(path)
    assert (loaded.predict(X_test) == model.predict(X_test)).all()
