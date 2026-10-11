
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "api_model.joblib"

app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting machine failures.",
    version="1.0.0",
)

pipeline = None

if MODEL_PATH.exists():
    pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print(f"Model file not found: {MODEL_PATH}")


class PredictionRequest(BaseModel):
    features: dict[str, Any]


@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API is running.",
        "model_loaded": pipeline is not None,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model is available.",
        )

    expected_features = getattr(pipeline, "feature_names_in_", None)

    if expected_features is None:
        raise HTTPException(
            status_code=500,
            detail="Could not determine the model's input features.",
        )

    missing = [
        feature for feature in expected_features
        if feature not in request.features
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing},
        )

    input_data = pd.DataFrame([{
        feature: request.features[feature]
        for feature in expected_features
    }])

    try:
        prediction = int(pipeline.predict(input_data)[0])

        probability_failure = None
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(input_data)[0]
            classes = list(pipeline.classes_)

            if 1 in classes:
                probability_failure = float(
                    probabilities[classes.index(1)]
                )

        return {
            "prediction": prediction,
            "label": (
                "Machine failure"
                if prediction == 1
                else "No machine failure"
            ),
            "probability_failure": probability_failure,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(exc)}",
        )