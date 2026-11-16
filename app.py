"""Atelier 4 : API REST FastAPI exposant la fonction predict()."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_pipeline import FEATURES, load_model, predict

MODEL_PATH = "classifier.joblib"

app = FastAPI(title="Churn API", description="Prediction du depart des clients")
_cache = {}


class CustomerFeatures(BaseModel):
    """Caracteristiques d'un client (Gender : 0 = Female, 1 = Male)."""

    CreditScore: int
    Gender: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "CreditScore": 850,
                "Gender": 0,
                "Age": 43,
                "Tenure": 2,
                "Balance": 125510.82,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 79084.10,
            }
        }
    }


def get_model():
    """Charge le modele depuis le disque (une seule fois) ou leve une erreur claire."""
    if "model" not in _cache:
        try:
            _cache["model"] = load_model(MODEL_PATH)
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail=f"Modele introuvable : {MODEL_PATH}. Entrainez-le d'abord.",
            )
    return _cache["model"]


@app.get("/")
def root():
    """Verifie que l'API est en ligne."""
    return {"status": "ok", "docs": "/docs"}


@app.post("/predict")
def predict_route(customer: CustomerFeatures):
    """Retourne la prediction de depart pour un client."""
    model = get_model()
    values = [getattr(customer, name) for name in FEATURES]
    try:
        prediction, probability = predict(model, values)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur de prediction : {exc}")
    return {
        "prediction": prediction,
        "label": "Churn" if prediction == 1 else "No churn",
        "probability": round(probability, 4),
    }
