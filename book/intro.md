# Heart Disease MLOps — Proyecto Integrador

## Descripción

Proyecto de **Machine Learning Operations (MLOps)** para la predicción de falla cardíaca.
Se desarrolla un flujo completo desde el análisis exploratorio hasta el despliegue del modelo.

## Dataset

[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) — 918 registros con 11 características clínicas.

## Contenido

1. **Preprocesamiento y Data Leakage** — Demostración del impacto de la fuga de datos y cómo evitarla con `Pipeline`.
2. **Modelado con Pipeline + GridSearchCV** — Entrenamiento de 5 modelos, comparación por AUC/Accuracy, exportación del mejor modelo.
3. **MLOps en Producción** — Despliegue con FastAPI, Docker, Kubernetes, CI/CD con GitHub Actions, y monitoreo con Evidently.

## Etapas del Proyecto

| Etapa | Descripción |
|-------|-------------|
| 0. Estructura | Estructura modular de carpetas |
| 1. Análisis & Preprocesamiento | EDA, detección de data leakage |
| 2. Entrenamiento Seguro | Pipeline + GridSearchCV con 5 modelos |
| 3. Despliegue Local | API REST con FastAPI + Docker |
| 4. Orquestación | Manifiestos de Kubernetes |
| 5. CI/CD | GitHub Actions (lint + tests) |
| 6. Monitoreo | Reporte de data drift con Evidently |

## Tecnologías

- Python, scikit-learn, pandas, matplotlib
- FastAPI + Uvicorn
- Docker + Kubernetes
- GitHub Actions
- Evidently (monitoreo)
- Jupyter Book (documentación)
