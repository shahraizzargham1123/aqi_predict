"""
Turns raw hourly readings into the daily feature table we actually train on.

The flow is: pull raw data -> collapse to one row per day -> add time and
derived features -> attach the next-3-days AQI targets -> save to the feature
store.

Run it two ways:
    python feature_pipeline/build_features.py --backfill   # seed ~90 days
    python feature_pipeline/build_features.py              # hourly top-up
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import config
from utils import feature_store
from data_pipeline.fetch_data import fetch_history, fetch_recent

# Columns we average when collapsing hours into a day.
_DAILY_MEAN_COLS = config.AIR_QUALITY_VARS + config.WEATHER_VARS + ["aqi"]


def to_daily(hourly_df):
    """Collapse hourly readings into daily averages, one row per calendar day."""
    df = hourly_df.copy()
    df["date"] = df["time"].dt.normalize()
    daily = df.groupby("date")[_DAILY_MEAN_COLS].mean().reset_index()
    daily["city"] = config.CITY_NAME
    return daily.sort_values("date").reset_index(drop=True)


def add_time_features(daily):
    """Calendar features. Day-of-week and month also get a cyclical encoding so
    the models see that December rolls back round to January."""
    daily = daily.copy()
    d = daily["date"].dt
    daily["day"] = d.day
    daily["month"] = d.month
    daily["day_of_week"] = d.dayofweek
    daily["day_of_year"] = d.dayofyear
    daily["is_weekend"] = (d.dayofweek >= 5).astype(int)

    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)
    daily["dow_sin"] = np.sin(2 * np.pi * daily["day_of_week"] / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * daily["day_of_week"] / 7)
    return daily


def add_derived_features(daily):
    """Lags, rolling stats and the AQI change rate the brief asked for. These
    are what let a plain regressor pick up on momentum and recent trend."""
    daily = daily.copy()

    # Yesterday-to-today change, and the same as a percentage.
    daily["aqi_change"] = daily["aqi"].diff()
    daily["aqi_change_rate"] = daily["aqi"].pct_change()

    # How dirty the last few days have been.
    for lag in (1, 2, 3):
        daily[f"aqi_lag_{lag}"] = daily["aqi"].shift(lag)
    daily["aqi_roll_mean_3"] = daily["aqi"].rolling(3).mean()
    daily["aqi_roll_mean_7"] = daily["aqi"].rolling(7).mean()
    daily["aqi_roll_std_3"] = daily["aqi"].rolling(3).std()

    # A couple of weather lags help too, pollution tends to trail the weather.
    daily["temp_lag_1"] = daily["temperature_2m"].shift(1)
    daily["wind_lag_1"] = daily["wind_speed_10m"].shift(1)
    return daily


def add_targets(daily):
    """The thing we predict: AQI 1, 2 and 3 days into the future."""
    daily = daily.copy()
    for h in range(1, config.FORECAST_HORIZON + 1):
        daily[f"aqi_next_{h}"] = daily["aqi"].shift(-h)
    return daily


def build_feature_frame(hourly_df, drop_incomplete=True):
    """Run the whole transform. With drop_incomplete we throw away rows that are
    missing lags or targets; for the live top-up we keep the latest day even
    though its future targets aren't known yet."""
    daily = to_daily(hourly_df)
    daily = add_time_features(daily)
    daily = add_derived_features(daily)
    daily = add_targets(daily)

    target_cols = [f"aqi_next_{h}" for h in range(1, config.FORECAST_HORIZON + 1)]
    # The earliest rows never have their lag/rolling windows filled.
    daily = daily.dropna(subset=["aqi_lag_3", "aqi_roll_mean_7"])

    if drop_incomplete:
        daily = daily.dropna(subset=target_cols)

    return daily.reset_index(drop=True)


def run_backfill():
    """Seed the feature store with the last ~90 days of history."""
    end = date.today() - timedelta(days=3)          # archive lags a few days
    start = end - timedelta(days=config.BACKFILL_DAYS)
    print(f"Backfilling {config.CITY_NAME} from {start} to {end}...")

    hourly = fetch_history(start.isoformat(), end.isoformat())
    features = build_feature_frame(hourly, drop_incomplete=True)
    print(f"Built {len(features)} daily feature rows.")
    feature_store.save_features(features)


def run_recent():
    """Hourly top-up. Grab a short window so lags/rolling stats are valid, keep
    the most recent rows even if their targets aren't known yet."""
    print(f"Updating {config.CITY_NAME} with recent data...")
    hourly = fetch_recent(past_days=14)
    features = build_feature_frame(hourly, drop_incomplete=False)
    print(f"Built {len(features)} daily feature rows.")
    feature_store.save_features(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build AQI features.")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Pull ~90 days of history instead of the recent rolling window.",
    )
    args = parser.parse_args()

    if args.backfill:
        run_backfill()
    else:
        run_recent()
