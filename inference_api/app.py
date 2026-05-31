"""
A small FastAPI service that serves the AQI forecast.

It loads the registered model once at startup and exposes a couple of JSON
endpoints: a health check, the 3-day forecast, and a plain-text alert summary.
The dashboard can call this, or you can just hit it with curl.

Run it with:
    uvicorn inference_api.app:app --reload
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import config
from utils.predict import latest_forecast

app = FastAPI(
    title="Pearls AQI Predictor",
    description=f"3-day AQI forecast for {config.CITY_NAME}",
    version="1.0.0",
)


@app.get("/")
def root():
    """Quick health check so you can tell the service is up."""
    return {"status": "ok", "city": config.CITY_NAME, "service": "aqi-predictor"}


@app.get("/predict")
def predict():
    """The full 3-day forecast with categories and hazard flags."""
    try:
        return latest_forecast()
    except FileNotFoundError as exc:
        # No model or features yet, point the caller at what to run.
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/alert")
def alert():
    """A trimmed-down view that only cares about hazardous days."""
    result = latest_forecast()
    hazardous_days = [d for d in result["forecast"] if d["hazardous"]]
    return {
        "city": result["city"],
        "any_hazardous": result["any_hazardous"],
        "threshold": config.HAZARDOUS_AQI_THRESHOLD,
        "hazardous_days": hazardous_days,
        "message": (
            f"Heads up: {len(hazardous_days)} of the next "
            f"{config.FORECAST_HORIZON} days look hazardous in {result['city']}."
            if hazardous_days else
            f"Air quality in {result['city']} stays below the hazardous line "
            f"for the next {config.FORECAST_HORIZON} days."
        ),
    }
