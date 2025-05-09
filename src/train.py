import mlflow
import mlflow.sklearn
import yaml
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models import infer_signature

import pandas as pd
from src.preprocessing import load_and_preprocess_data


def main(config_path):
    # -------------------------
    # 1. Cargar configuración YAML
    # -------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # -------------------------
    # 2. Preprocesar datos
    # -------------------------
    # Incluye limpieza, codificación y escalado
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config["data_path"])

    # -------------------------
    # 3. Crear modelo con parámetros del config
    # -------------------------
    model = LogisticRegression(**config["model_params"])

    # -------------------------
    # 4. Entrenar el modelo
    # -------------------------
    model.fit(X_train, y_train)

    # -------------------------
    # 5. Evaluar el modelo
    # -------------------------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # -------------------------
    # 6. Configurar MLflow Tracking
    # -------------------------
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # -------------------------
    # 7. Registrar modelo y métricas en MLflow
    # -------------------------
    with mlflow.start_run():
        # Log de parámetros y métricas
        mlflow.log_params(config["model_params"])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Ejemplo de entrada y firma del modelo (para trazabilidad)
        input_example = pd.DataFrame(X_train[:5])
        signature = infer_signature(X_train, model.predict(X_train))

        # Registro del modelo con firma e input de ejemplo
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

        print(f"[INFO] Modelo entrenado correctamente. Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")


# -------------------------
# EJECUCIÓN DESDE CONSOLA
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento y registro de un modelo de clasificación con MLflow")
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al archivo de configuración YAML")
    args = parser.parse_args()

    main(args.config)
