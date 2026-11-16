"""Atelier 3 : pipeline d'automatisation avec Prefect."""

import argparse
import subprocess
import sys

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from model_pipeline import (
    evaluate_model,
    load_model,
    log_experiment,
    prepare_data,
    save_model,
    train_model,
)

FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py", "app.py"]
DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"


def run_cmd(cmd):
    """Execute une commande avec l'interpreteur du venv et leve une erreur si echec."""
    cmd = [sys.executable, "-m"] + cmd
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Echec de la commande : {' '.join(cmd)}")


# ---------- Task : recuperation du projet (Git distant) ----------
@task(name="fetch_project")
def fetch_project():
    """Recupere la derniere version du projet depuis le depot distant."""
    remotes = subprocess.run(
        ["git", "remote"], capture_output=True, text=True, check=False
    ).stdout.split()
    if "origin" not in remotes:
        print("Pas de depot distant 'origin' : etape ignoree.")
        return
    result = subprocess.run(
        ["git", "pull", "--ff-only", "origin", "main"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("Echec de git pull")


# ---------- Tasks : code ----------
@task(name="install_dependencies")
def install_dependencies():
    run_cmd(["pip", "install", "-r", "requirements.txt"])


@task(name="format_code")
def format_code():
    run_cmd(["black"] + FILES)


@task(name="check_quality")
def check_quality():
    run_cmd(["flake8", "--max-line-length", "88"] + FILES)


@task(name="check_security")
def check_security():
    run_cmd(["bandit", "-q", "-ll"] + FILES)


@task(name="run_tests")
def run_tests():
    run_cmd(["pytest", "-q", "test_model_pipeline.py"])


# ---------- Tasks : data / modele ----------
@task(name="prepare_data", cache_policy=NO_CACHE)
def prepare_task():
    return prepare_data(DATA_PATH)


@task(name="train_model", cache_policy=NO_CACHE)
def train_task(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="save_model", cache_policy=NO_CACHE)
def save_task(model):
    save_model(model, MODEL_PATH)


@task(name="load_model", cache_policy=NO_CACHE)
def load_task():
    return load_model(MODEL_PATH)


@task(name="evaluate_model", cache_policy=NO_CACHE)
def evaluate_task(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


@task(name="start_api")
def start_api():
    """Lance l'API FastAPI (bloquant, arret avec Ctrl+C)."""
    run_cmd(["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"])


@task(name="track_experiment", cache_policy=NO_CACHE)
def track_task(model, X_test, y_test):
    log_experiment(model, X_test, y_test)


@task(name="start_mlflow")
def start_mlflow():
    """Lance l'interface MLflow (bloquant, arret avec Ctrl+C)."""
    run_cmd(
        [
            "mlflow",
            "ui",
            "--backend-store-uri",
            "sqlite:///mlflow.db",
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
        ]
    )


# ---------- Flows ----------
@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="train")
def train_flow():
    X_train, X_test, y_train, y_test = prepare_task()
    model = train_task(X_train, y_train)
    save_task(model)
    track_task(model, X_test, y_test)


@flow(name="evaluate")
def evaluate_flow():
    X_train, X_test, y_train, y_test = prepare_task()
    model = load_task()
    evaluate_task(model, X_test, y_test)


@flow(name="all")
def all_flow():
    fetch_project()
    code_flow()
    X_train, X_test, y_train, y_test = prepare_task()
    model = train_task(X_train, y_train)
    save_task(model)
    evaluate_task(model, X_test, y_test)
    track_task(model, X_test, y_test)


@flow(name="mlflow")
def mlflow_flow():
    start_mlflow()


@flow(name="api")
def api_flow():
    start_api()


FLOWS = {
    "all": all_flow,
    "train": train_flow,
    "evaluate": evaluate_flow,
    "code": code_flow,
    "api": api_flow,
    "mlflow": mlflow_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect - Churn")
    parser.add_argument("--flow", choices=list(FLOWS), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
