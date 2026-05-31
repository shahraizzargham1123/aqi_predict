"""
Shared prediction logic.

Both the API and the dashboard need the same thing: take the most recent day of
features, run the model, and turn three raw numbers into a friendly 3-day
forecast. Keeping it here means they can't drift apart.
"""

from datetime import timedelta

import pandas as pd

from utils import config
from utils import feature_store
from utils import registry
from utils.aqi import aqi_category


def latest_forecast():
    """Predict AQI for the next FORECAST_HORIZON days from the latest features."""
    bundle = registry.load_model()
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    df = feature_store.read_features().sort_values("date").reset_index(drop=True)
    latest = df.iloc[-1]
    base_date = pd.to_datetime(latest["date"])

    X = latest[feature_names].to_frame().T
    preds = model.predict(X)[0]

    forecast = []
    for i, value in enumerate(preds, start=1):
        aqi = round(float(value))
        forecast.append({
            "horizon": i,
            "date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            "aqi": aqi,
            "category": aqi_category(aqi),
            "hazardous": aqi >= config.HAZARDOUS_AQI_THRESHOLD,
        })

    current_aqi = round(float(latest["aqi"]))
    return {
        "city": config.CITY_NAME,
        "model": bundle.get("model_label", "unknown"),
        "based_on": base_date.strftime("%Y-%m-%d"),
        "current_aqi": current_aqi,
        "current_category": aqi_category(current_aqi),
        "forecast": forecast,
        "any_hazardous": any(day["hazardous"] for day in forecast),
        "metrics": bundle.get("metrics", {}),
    }


def recent_history(days=30):
    """Last N days of observed AQI, handy for plotting the trend."""
    df = feature_store.read_features().sort_values("date")
    cols = ["date", "aqi"]
    return df[cols].tail(days).reset_index(drop=True)
