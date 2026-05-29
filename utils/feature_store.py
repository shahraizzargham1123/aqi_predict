"""
Thin layer over the Hopsworks feature store.

The idea is the rest of the code never has to care *where* the features live.
If a Hopsworks API key is configured we read/write there. If it isn't (or the
library can't be imported), we quietly fall back to a local CSV under data/ so
the pipeline still runs end to end while developing or demoing offline.
"""

from pathlib import Path

import pandas as pd

from utils import config

# Where the offline copy lives when Hopsworks isn't available.
_LOCAL_DIR = Path(__file__).resolve().parents[1] / "data"
_LOCAL_FILE = _LOCAL_DIR / "features.csv"


def _hopsworks_available():
    """We only use Hopsworks if there's a key set and the library is importable."""
    if not config.HOPSWORKS_API_KEY:
        return False
    try:
        import hopsworks  # noqa: F401
        return True
    except ImportError:
        return False


def _login():
    import hopsworks
    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT,
    )
    return project


def save_features(df):
    """
    Upsert the given feature rows. On Hopsworks this inserts into the feature
    group (which dedupes on the primary key); locally it merges into the CSV.
    """
    if df.empty:
        print("Nothing to save, the feature frame is empty.")
        return

    if _hopsworks_available():
        project = _login()
        fs = project.get_feature_store()
        fg = fs.get_or_create_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
            primary_key=["city", "date"],
            event_time="date",
            description="Daily AQI features and 3-day-ahead targets for "
                        f"{config.CITY_NAME}",
        )
        fg.insert(df, write_options={"wait_for_job": True})
        print(f"Wrote {len(df)} rows to Hopsworks feature group "
              f"'{config.FEATURE_GROUP_NAME}'.")
    else:
        _save_local(df)


def read_features():
    """Pull every stored feature row back as a DataFrame, sorted by date."""
    if _hopsworks_available():
        project = _login()
        fs = project.get_feature_store()
        fg = fs.get_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
        )
        df = fg.read()
    else:
        if not _LOCAL_FILE.exists():
            raise FileNotFoundError(
                f"No local feature file at {_LOCAL_FILE}. Run the feature "
                "pipeline first (or set HOPSWORKS_API_KEY to use Hopsworks)."
            )
        df = pd.read_csv(_LOCAL_FILE, parse_dates=["date"])

    return df.sort_values("date").reset_index(drop=True)


def _save_local(df):
    _LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    if _LOCAL_FILE.exists():
        existing = pd.read_csv(_LOCAL_FILE, parse_dates=["date"])
        combined = pd.concat([existing, df], ignore_index=True)
        # Last write wins for a given city/date so reruns stay clean.
        combined = combined.drop_duplicates(subset=["city", "date"], keep="last")
    else:
        combined = df
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(_LOCAL_FILE, index=False)
    print(f"Wrote {len(df)} rows to local feature file {_LOCAL_FILE} "
          f"({len(combined)} rows total).")
