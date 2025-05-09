# MLOps: Heart Disease Prediction

Este proyecto automatiza un pipeline de Machine Learning usando GitHub Actions y MLflow.

## 📦 Estructura

- `src/`: Scripts de entrenamiento, predicción y preprocesamiento
- `data/`: Dataset usado (`HeartDiseaseTrain-Test.csv`)
- `config.yaml`: Configuración del modelo
- `Makefile`: Comandos automatizados
- `.github/workflows/`: CI/CD pipeline

## 🚀 Cómo ejecutar localmente

```bash
make install
make train
make test
```

## 🔮 Predicción

```bash
python src/predict.py \
  --model-uri models:/nombre-modelo/1 \
  --input-csv data/HeartDiseaseTrain-Test.csv \
  --output-csv predicciones.csv
```

## 📊 MLflow

Lanza la interfaz con:

```bash
mlflow ui
```

## 🔁 CI/CD con GitHub Actions

Este repositorio incluye `.github/workflows/ml.yml` para ejecutar:
- Instalación
- Entrenamiento
- Publicación de artefactos

## ✅ Requisitos

- Python 3.8 o superior
- MLflow
- GitHub Actions habilitado