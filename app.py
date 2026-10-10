
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query


# --------------------------------------------------
# 1. Initialize application
# --------------------------------------------------

app = FastAPI(
    title="Study Score Prediction API",
    description="Predict scores using a trained Linear Regression model.",
    version="1.0.0",
)


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model artifact not found: {MODEL_PATH}. "
        "Run train.py first to generate the model."
    )

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 3. Health endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "ML Model API is running",
        "model": "LinearRegression",
        "status": "healthy",
    }


# --------------------------------------------------
# 4. Model metrics endpoint
# --------------------------------------------------

@app.get("/metrics")
def get_metrics():
    metrics_path = BASE_DIR / "artifacts" / "metrics.json"

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model metrics not found. Run train.py first.",
        )

    import json

    with metrics_path.open() as file:
        metrics = json.load(file)

    return metrics


# --------------------------------------------------
# 5. Prediction endpoint
# --------------------------------------------------

@app.get("/predict")
def predict(
    hours: float = Query(..., gt=0, le=24)
):
    try:
        prediction = model.predict(
            np.array([[hours]], dtype=float)
        )

        return {
            "hours_studied": hours,
            "predicted_score": round(float(prediction[0]), 2),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

