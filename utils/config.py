"""
Central place for all the knobs we tweak across the project.

Everything that's environment specific (API keys, project names) is read from
the environment / a local .env file, while the stuff that rarely changes
(the city we're forecasting, which API we hit) lives here as plain constants.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env if there is one. On GitHub Actions the
# values come straight from the environment instead, so this is a no-op there.
load_dotenv()


# The city we're forecasting for. Swapping cities is just a matter of changing
# these two numbers and the label.
CITY_NAME = "Lahore"
LATITUDE = 31.5204
LONGITUDE = 74.3587
TIMEZONE = "Asia/Karachi"

# How far back we backfill when first seeding the feature store.
BACKFILL_DAYS = 90

# How many days ahead we forecast. The targets and the dashboard both lean on
# this, so keep it in one spot.
FORECAST_HORIZON = 3

# Open-Meteo endpoints. No API key needed, which is exactly why we picked it.
# The "archive" hosts are the historical ones used for the backfill.
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Pollutants and weather variables we pull. Order doesn't matter to the API but
# keeping them listed makes it obvious what we depend on.
AIR_QUALITY_VARS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
]

# Hopsworks bits. The API key is a secret so it stays in the environment.
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqi_predict")

# Names we use inside Hopsworks. Bumping the version is how we'd evolve the
# schema later without clobbering old data.
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "aqi_forecaster"

# AQI is considered unhealthy past this point (US EPA "Unhealthy" band starts at
# 151). The API and dashboard use this to decide when to raise an alert.
HAZARDOUS_AQI_THRESHOLD = 151
