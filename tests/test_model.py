"""Tests básicos para el proyecto Heart Disease MLOps."""


def test_model_file_exists():
    """Verifica que el modelo exportado existe."""
    import os
    assert os.path.exists("app/model.joblib")


def test_model_loads():
    """Verifica que el modelo se carga correctamente."""
    import joblib
    model = joblib.load("app/model.joblib")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_model_predicts():
    """Verifica que el modelo genera predicciones."""
    import joblib
    import numpy as np
    model = joblib.load("app/model.joblib")
    # 17 features (después de one-hot encoding)
    X = np.zeros((1, model.n_features_in_))
    pred = model.predict(X)
    proba = model.predict_proba(X)
    assert pred.shape == (1,)
    assert proba.shape[0] == 1
    assert 0 <= proba[0][1] <= 1
