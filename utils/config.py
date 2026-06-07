"""
Central place for all the knobs I tweak across the project.

Everything that's environment specific (API keys, project names) is read from
the environment / a local .env file, while the stuff that rarely changes
(the city I'm forecasting, which API I hit) lives here as plain constants.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env if there is one. On GitHub Actions the
# values come straight from the environment instead, so this is a no-op there.
load_dotenv()


# The city I'm forecasting for. Swapping cities is just a matter of changing
# these two numbers and the label.
CITY_NAME = "Lahore"
LATITUDE = 31.5204
LONGITUDE = 74.3587
TIMEZONE = "Asia/Karachi"

# How far back I backfill when first seeding the feature store.
BACKFILL_DAYS = 90

# How many days ahead I forecast. The targets and the dashboard both lean on
# this, so keep it in one spot.
FORECAST_HORIZON = 3

# How much worse than the current model a freshly trained one is allowed to be
# and still get promoted. This is the guardrail: if a day's retrain comes out
# noticeably worse than what's already serving, I keep the old one instead.
PROMOTION_TOLERANCE = 0.20

# Open-Meteo endpoints. No API key needed, which is exactly why I picked it.
# The "archive" hosts are the historical ones used for the backfill.
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Pollutants and weather variables I pull. Order doesn't matter to the API but
# keeping them listed makes it obvious what I depend on.
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

# Names I use inside Hopsworks. Bumping the version is how I'd evolve the
# schema later without clobbering old data.
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "aqi_forecaster"

# AQI is considered unhealthy past this point (US EPA "Unhealthy" band starts at
# 151). The API and dashboard use this to decide when to raise an alert.
HAZARDOUS_AQI_THRESHOLD = 151
