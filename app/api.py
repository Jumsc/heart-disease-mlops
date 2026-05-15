from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo entrenado (pipeline completo con scaler + clasificador)
model = joblib.load("app/model.joblib")
app = FastAPI(title="Heart Disease Prediction API", version="1.0")


class Input(BaseModel):
    features: list

    class Config:
        json_schema_extra = {
            "example": {
                "features": [
                    52, 140, 230, 0, 150, 1.2,
                    1, 0, 0, 1, 0, 0,
                    1, 0, 0, 1, 0
                ]
            }
        }


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API - POST /predict con features"}


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
