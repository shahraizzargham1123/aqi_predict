"""
Pulls raw pollutant + weather readings from Open-Meteo and stitches them into a
single hourly table, with the AQI worked out for every row.

There are two ways I use this:
  - fetch_history(start, end)  -> the big historical pull used for the backfill
  - fetch_recent(past_days)    -> the small rolling window the hourly job uses

Open-Meteo is free and needs no API key, which is the whole reason I went with
it. The catch is the historical weather ("archive") feed lags real time by a few
days, so for recent data I ask the regular forecast feed for its "past_days"
instead.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

# Make sure I can import the project packages when run as a plain script.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import config
from utils.aqi import compute_aqi

# Open-Meteo occasionally throws a transient 5xx or rate-limits me. These are the
# codes worth waiting out and retrying rather than giving up on.
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def _get_json(url, params, retries=4, backoff=2.0):
    """GET some JSON, retrying with backoff on the kind of hiccups that sort
    themselves out (timeouts, dropped connections, transient 5xx). A real client
    error (bad params, etc.) still raises straight away."""
    for attempt in range(retries):
        last = attempt == retries - 1
        try:
            response = requests.get(url, params=params, timeout=60)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if last:
                raise
            wait = backoff * (2 ** attempt)
            print(f"Open-Meteo connection issue ({exc}); retrying in {wait:.0f}s")
            time.sleep(wait)
            continue

        if response.status_code in _TRANSIENT_STATUS and not last:
            wait = backoff * (2 ** attempt)
            print(f"Open-Meteo returned {response.status_code}; retrying in "
                  f"{wait:.0f}s")
            time.sleep(wait)
            continue

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

    # Rows where I couldn't compute an AQI are useless downstream.
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
    Rolling recent window for the hourly job. I grab a handful of past days so
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
