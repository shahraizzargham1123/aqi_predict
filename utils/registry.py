"""
Saving and loading the trained model, again with a Hopsworks-or-local split.

We bundle everything the predictor needs into one dict (the fitted estimator,
the exact feature column order, the metrics and a little metadata) and persist
that with joblib. The API and the dashboard both load through here, so they
don't need to know whether the model came from Hopsworks or a local file.
"""

import json
import tempfile
from pathlib import Path

import joblib

from utils import config

_LOCAL_DIR = Path(__file__).resolve().parents[1] / "models"
_LOCAL_MODEL = _LOCAL_DIR / "model.pkl"
_LOCAL_METRICS = _LOCAL_DIR / "metrics.json"


def _hopsworks_available():
    if not config.HOPSWORKS_API_KEY:
        return False
    try:
        import hopsworks  # noqa: F401
        return True
    except ImportError:
        return False


def _login():
    import hopsworks
    return hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT,
    )


def save_model(estimator, feature_names, metrics, model_label):
    """Persist the winning model. Always writes a local copy; also pushes to the
    Hopsworks model registry when it's configured."""
    bundle = {
        "model": estimator,
        "feature_names": list(feature_names),
        "metrics": metrics,
        "model_label": model_label,
        "horizon": config.FORECAST_HORIZON,
        "city": config.CITY_NAME,
    }

    _LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, _LOCAL_MODEL)
    with open(_LOCAL_METRICS, "w") as f:
        json.dump({"model": model_label, **metrics}, f, indent=2)
    print(f"Saved model locally to {_LOCAL_MODEL}")

    if _hopsworks_available():
        project = _login()
        mr = project.get_model_registry()
        # Hopsworks registers a whole directory, so drop the bundle into a temp
        # folder and hand that over.
        with tempfile.TemporaryDirectory() as tmp:
            joblib.dump(bundle, Path(tmp) / "model.pkl")
            registry_metrics = {
                k: float(v) for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            model = mr.python.create_model(
                name=config.MODEL_NAME,
                metrics=registry_metrics,
                description=f"{model_label} forecasting {config.CITY_NAME} AQI "
                            f"{config.FORECAST_HORIZON} days ahead",
            )
            model.save(tmp)
        print(f"Registered model '{config.MODEL_NAME}' in Hopsworks.")


def load_model():
    """Load the latest model bundle. Prefers Hopsworks, falls back to local."""
    if _hopsworks_available():
        project = _login()
        mr = project.get_model_registry()
        model = mr.get_best_model(
            name=config.MODEL_NAME,
            metric="rmse",
            direction="min",
        )
        download_dir = model.download()
        return joblib.load(Path(download_dir) / "model.pkl")

    if not _LOCAL_MODEL.exists():
        raise FileNotFoundError(
            f"No model at {_LOCAL_MODEL}. Run the training pipeline first."
        )
    return joblib.load(_LOCAL_MODEL)
