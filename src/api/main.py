
import logging
import math
import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bbca-api")

# MLflow configuration
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-server:5055"
)

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/bbca-direction-classifier/1"
)

# Exact feature order from the Production model signature
FEATURE_COLUMNS = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "BB_Upper",
    "BB_Lower",
]


class FeatureRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Close: float
    High: float
    Low: float
    Open: float
    Volume: int
    RSI_14: float
    MACD: float
    MACD_Signal: float
    BB_Upper: float
    BB_Lower: float


class PredictionRequest(BaseModel):
    instances: list[FeatureRow] = Field(min_length=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Production model when the API starts."""

    logger.info("Connecting to MLflow: %s", TRACKING_URI)
    logger.info("Loading model: %s", MODEL_URI)

    mlflow.set_tracking_uri(TRACKING_URI)

    try:
        app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
        logger.info("Production model loaded successfully.")
    except Exception:
        logger.exception("Failed to load Production model.")
        raise

    yield

    app.state.model = None
    logger.info("API shutdown completed.")


app = FastAPI(
    title="BBCA Stock Prediction API",
    description=(
        "Inference API for BBCA price direction "
        "using the Production model registered in MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "application": "BBCA Stock Prediction API",
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/health")
def health(request: Request):
    model_loaded = getattr(request.app.state, "model", None) is not None

    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Production model is not loaded."
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_uri": MODEL_URI,
        "tracking_uri": TRACKING_URI,
    }


@app.get("/model-info")
def model_info():
    return {
        "registered_model": "bbca-direction-classifier",
        "version": 1,
        "stage": "Production",
        "model_uri": MODEL_URI,
        "tracking_uri": TRACKING_URI,
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "output_mapping": {
            "0": "Non-Up",
            "1": "Up",
        },
    }


@app.post("/predict")
def predict(payload: PredictionRequest, request: Request):
    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Production model is not available."
        )

    rows = [instance.model_dump() for instance in payload.instances]

    # Validate finite numeric values.
    for index, row in enumerate(rows):
        for feature, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid numeric value at row {index}: {feature}"
                )

    # Preserve the exact feature order and Volume integer type.
    dataframe = pd.DataFrame(
        rows,
        columns=FEATURE_COLUMNS
    )

    try:
        predictions = model.predict(dataframe)
        predictions = [int(value) for value in predictions]

    except Exception:
        logger.exception("Inference failed.")
        raise HTTPException(
            status_code=500,
            detail="Model inference failed."
        )

    labels = [
        "Up" if value == 1 else "Non-Up"
        for value in predictions
    ]

    return {
        "status": "success",
        "model_uri": MODEL_URI,
        "tracking_uri": TRACKING_URI,
        "input_samples": len(rows),
        "feature_count": len(FEATURE_COLUMNS),
        "predictions": predictions,
        "prediction_labels": labels,
    }
