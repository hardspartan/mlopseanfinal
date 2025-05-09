import mlflow
import mlflow.sklearn
import pandas as pd
import argparse

def predict(model_uri, input_csv, output_csv=None):
    # 1. Configura el URI del tracking server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # 2. Carga el modelo registrado en MLflow
    model = mlflow.sklearn.load_model(model_uri)
    print(f"[INFO] Modelo cargado desde: {model_uri}")

    # 3. Carga los datos a predecir
    data = pd.read_csv(input_csv)

    # 4. Realiza las predicciones
    predictions = model.predict(data)

    # 5. Agrega las predicciones al DataFrame original
    data["prediction"] = predictions

    # 6. Muestra por consola
    print(data[["prediction"]])

    # 7. (Opcional) Guarda el resultado
    if output_csv:
        data.to_csv(output_csv, index=False)
        print(f"[INFO] Resultados guardados en: {output_csv}")

# -------------------------
# EJECUCIÓN POR CONSOLA
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicción con modelo de MLflow")
    parser.add_argument("--model-uri", required=True, help="Ruta del modelo (ej: models:/FINALMLOPS/1)")
    parser.add_argument("--input-csv", required=True, help="CSV con datos a predecir")
    parser.add_argument("--output-csv", help="Archivo donde guardar las predicciones")
    args = parser.parse_args()

    predict(args.model_uri, args.input_csv, args.output_csv)
