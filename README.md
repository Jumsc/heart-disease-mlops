# Heart Disease MLOps — Proyecto Integrador de Aprendizaje Automático

## Descripción
Proyecto de Machine Learning Operations (MLOps) para la predicción de falla cardíaca.
Se desarrolla un flujo completo desde el análisis exploratorio hasta el despliegue del
modelo con Docker, Kubernetes, CI/CD con GitHub Actions y monitoreo con Evidently.

## Dataset
[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) — 918 registros con 11 características clínicas.

## Estructura del Proyecto
```
heart-disease-mlops/
├── app/
│   └── api.py                          # API REST con FastAPI
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── notebooks/
│   ├── 1_model_leakage_demo.ipynb      # Etapa 1: EDA + Data Leakage
│   └── 2_model_pipeline_cv.ipynb       # Etapa 2: Pipeline + GridSearchCV
├── .github/
│   └── workflows/
│       └── ci.yml                      # Etapa 5: CI/CD
├── drift_report.html                   # Etapa 6: Monitoreo (Evidently)
├── model.joblib                        # Modelo exportado
├── heart.csv                           # Dataset
└── README.md
```

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

## Modelos Evaluados
- SVC (Support Vector Classifier)
- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors
- Gradient Boosting Classifier

## Cómo Ejecutar

### 1. Notebooks
```bash
cd notebooks/
jupyter notebook
```

### 2. Docker
```bash
docker build -t heart-api -f docker/Dockerfile .
docker run -p 8000:8000 heart-api
```

### 3. Kubernetes (Minikube)
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get svc
```

### 4. Probar la API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [52, 140, 230, 0, 150, 1.2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]}'
```

## Tecnologías
- Python, scikit-learn, pandas, matplotlib
- FastAPI + Uvicorn
- Docker + Kubernetes
- GitHub Actions
- Evidently (monitoreo)
- Jupyter Notebooks

## Autor
Proyecto Integrador — Curso de Machine Learning (Pipelines)
