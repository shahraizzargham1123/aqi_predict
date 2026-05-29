"""
Pulls raw pollutant + weather readings from Open-Meteo and stitches them into a
single hourly table, with the AQI worked out for every row.

There are two ways we use this:
  - fetch_history(start, end)  -> the big historical pull used for the backfill
  - fetch_recent(past_days)    -> the small rolling window the hourly job uses

Open-Meteo is free and needs no API key, which is the whole reason we went with
it. The catch is the historical weather ("archive") feed lags real time by a few
days, so for recent data we ask the regular forecast feed for its "past_days"
instead.
"""

import sys
from pathlib import Path

import pandas as pd
import requests

# Make sure we can import the project packages when run as a plain script.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import config
from utils.aqi import compute_aqi


def _get_json(url, params):
    """One small wrapper so every call gets the same timeout and error handling."""
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def _hourly_frame(payload):
    """Turn Open-Meteo's {"hourly": {...}} block into a tidy DataFrame."""
    hourly = payload.get("hourly", {})
    frame = pd.DataFrame(hourly)
    if "time" in frame.columns:
        frame["time"] = pd.to_datetime(frame["time"])
    return frame


def _fetch_air_quality(params):
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": ",".join(config.AIR_QUALITY_VARS),
        "timezone": config.TIMEZONE,
        **params,
    }
    return _hourly_frame(_get_json(config.AIR_QUALITY_URL, params))


def _fetch_weather(url, params):
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": ",".join(config.WEATHER_VARS),
        "timezone": config.TIMEZONE,
        **params,
    }
    return _hourly_frame(_get_json(url, params))


def _combine(air_df, weather_df):
    """Merge pollutants + weather on the timestamp and add the AQI column."""
    if air_df.empty:
        return air_df

    merged = pd.merge(air_df, weather_df, on="time", how="inner")
    merged = merged.sort_values("time").reset_index(drop=True)

    # Work out the AQI row by row from whichever pollutants are present.
    merged["aqi"] = merged.apply(
        lambda row: compute_aqi(
            pm2_5=row.get("pm2_5"),
            pm10=row.get("pm10"),
            ozone=row.get("ozone"),
            nitrogen_dioxide=row.get("nitrogen_dioxide"),
            sulphur_dioxide=row.get("sulphur_dioxide"),
            carbon_monoxide=row.get("carbon_monoxide"),
        ),
        axis=1,
    )

    # Rows where we couldn't compute an AQI are useless downstream.
    merged = merged.dropna(subset=["aqi"]).reset_index(drop=True)
    return merged


def fetch_history(start_date, end_date):
    """
    Historical pull for the backfill. Dates are 'YYYY-MM-DD' strings. Weather
    comes from the archive feed, pollutants from the air-quality feed.
    """
    air = _fetch_air_quality({"start_date": start_date, "end_date": end_date})
    weather = _fetch_weather(
        config.WEATHER_ARCHIVE_URL,
        {"start_date": start_date, "end_date": end_date},
    )
    return _combine(air, weather)


def fetch_recent(past_days=7):
    """
    Rolling recent window for the hourly job. We grab a handful of past days so
    the feature pipeline has enough history to build lag/rolling features.
    """
    air = _fetch_air_quality({"past_days": past_days, "forecast_days": 1})
    weather = _fetch_weather(
        config.WEATHER_FORECAST_URL,
        {"past_days": past_days, "forecast_days": 1},
    )
    return _combine(air, weather)


if __name__ == "__main__":
    # Quick manual smoke test: pull the last few days and show the tail.
    df = fetch_recent(past_days=3)
    print(f"Fetched {len(df)} hourly rows for {config.CITY_NAME}")
    print(df[["time", "pm2_5", "pm10", "aqi"]].tail())
