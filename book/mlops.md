# MLOps en Producción

Este capítulo documenta las etapas de despliegue y operación del modelo de predicción de falla cardíaca.

## Etapa 3: API REST con FastAPI

El modelo entrenado se sirve a través de una API REST construida con FastAPI.

### Código de la API (`app/api.py`)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("app/model.joblib")
app = FastAPI(title="Heart Disease Prediction API", version="1.0")


class Input(BaseModel):
    features: list


@app.get("/")
def root():
    msg = "Heart Disease Prediction API - POST /predict"
    return {"message": msg}


@app.post("/predict")
def predict(data: Input):
    X = np.array(data.features).reshape(1, -1)
    proba = model.predict_proba(X)[0][1]
    return {
        "heart_disease_probability": round(float(proba), 4),
        "prediction": int(proba > 0.5)
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
```

### Ejemplo de uso

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [52, 140, 230, 0, 150, 1.2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]}'
```

Respuesta esperada:
```json
{
  "heart_disease_probability": 0.6823,
  "prediction": 1
}
```

---

## Etapa 4: Contenedorización con Docker

### Dockerfile

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Comandos

```bash
# Construir la imagen
docker build -t heart-api -f docker/Dockerfile .

# Ejecutar el contenedor
docker run -p 8000:8000 heart-api
```

---

## Etapa 5: Orquestación con Kubernetes

### Deployment (`k8s/deployment.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: heart-model
spec:
  replicas: 1
  selector:
    matchLabels:
      app: heart-model
  template:
    metadata:
      labels:
        app: heart-model
    spec:
      containers:
        - name: model
          image: <TU_USUARIO_DOCKER>/heart-api
          ports:
            - containerPort: 8000
```

### Service (`k8s/service.yaml`)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: heart-service
spec:
  selector:
    app: heart-model
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### Despliegue con Minikube

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get svc
```

---

## Etapa 6: CI/CD con GitHub Actions

El pipeline de integración continua se ejecuta en cada push y verifica:

1. **Lint** — Revisión de estilo con `flake8`
2. **Tests** — Pruebas unitarias con `pytest`

### Workflow (`.github/workflows/ci.yml`)

```yaml
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r docker/requirements.txt
          pip install flake8 pytest
      - name: Lint
        run: flake8 app/
      - name: Tests
        run: pytest tests/
```

---

## Etapa 7: Monitoreo con Evidently

Se genera un reporte de **data drift** para detectar cambios en la distribución de los datos entre entrenamiento y producción.

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_train, current_data=X_test)
report.save_html("drift_report.html")
```

El reporte HTML resultante permite visualizar qué features han cambiado su distribución, lo cual podría indicar que el modelo necesita reentrenarse.

---

## Arquitectura Completa

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Notebooks   │───▶│  model.joblib │───▶│  FastAPI     │
│  (Training)  │    │  (Pipeline)   │    │  (Serving)   │
└─────────────┘    └──────────────┘    └──────┬──────┘
                                              │
                                       ┌──────▼──────┐
                                       │   Docker     │
                                       │  Container   │
                                       └──────┬──────┘
                                              │
                                       ┌──────▼──────┐
                                       │ Kubernetes   │
                                       │  (K8s)       │
                                       └──────┬──────┘
                                              │
                   ┌──────────────┐    ┌──────▼──────┐
                   │  GitHub       │───▶│  Evidently   │
                   │  Actions CI  │    │  Monitoring  │
                   └──────────────┘    └─────────────┘
```
