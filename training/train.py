"""
Training pipeline: pull features from the store, race a few models against each
other, keep the best one and register it.

We compare Ridge, ElasticNet, Random Forest and XGBoost. Each one predicts all
three horizons at once (day+1, +2, +3) so a single model gives the whole 3-day
outlook. The winner is whichever has the lowest average RMSE across the three
days. There's room left to slot in a TensorFlow/LSTM model later, but the brief
for now is classical models, so that's what we ship.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import config
from utils import feature_store
from utils import registry

# XGBoost and SHAP are optional at runtime. If they're not installed the
# pipeline still works, it just skips that model / explanation.
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


TARGET_COLS = [f"aqi_next_{h}" for h in range(1, config.FORECAST_HORIZON + 1)]
NON_FEATURE_COLS = ["date", "city"] + TARGET_COLS

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def setup_logging():
    """Logs go to the console and to a timestamped file so each run is kept."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info("Logging to %s", log_file)
    return logging.getLogger(__name__)


def candidate_models():
    """The line-up. Linear models sit behind a scaler; the tree models don't
    need one. XGBoost only joins if it's installed."""
    models = {
        "ridge": Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "elasticnet": Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1,
        ),
    }
    if HAS_XGBOOST:
        # XGBoost isn't natively multi-output, so wrap one regressor per day.
        models["xgboost"] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
            )
        )
    return models


def evaluate(y_true, y_pred):
    """RMSE, MAE and R², averaged over the three horizons but also kept per-day
    so we can see which days are harder to call."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    per_day = {}
    rmses, maes, r2s = [], [], []
    for i, col in enumerate(TARGET_COLS):
        rmse = mean_squared_error(y_true[:, i], y_pred[:, i]) ** 0.5
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        per_day[col] = {"rmse": rmse, "mae": mae, "r2": r2}
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)

    return {
        "rmse": float(np.mean(rmses)),
        "mae": float(np.mean(maes)),
        "r2": float(np.mean(r2s)),
        "per_day": per_day,
    }


def time_split(df, test_fraction=0.2):
    """Time series, so no shuffling. The last chunk of days is the test set."""
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * (1 - test_fraction))
    return df.iloc[:split], df.iloc[split:]


def explain(best_estimator, X_train, X_test, feature_names, log):
    """Save feature importances. We use SHAP when it's available, and always
    fall back to the model's own importances/coefficients so there's something
    for the dashboard either way."""
    importance = None

    if HAS_SHAP:
        try:
            background = shap.sample(X_train, min(50, len(X_train)), random_state=42)
            # Explain the day+1 prediction; it's the headline number and keeps
            # the explainer single-output and fast.
            explainer = shap.Explainer(
                lambda data: best_estimator.predict(data)[:, 0], background
            )
            shap_values = explainer(X_test)
            importance = np.abs(shap_values.values).mean(axis=0)
            log.info("Computed SHAP importances for the day+1 forecast.")
        except Exception as exc:  # SHAP can be fiddly, never let it kill training
            log.warning("SHAP failed (%s), falling back to native importances.", exc)

    if importance is None:
        importance = _native_importance(best_estimator, len(feature_names))

    if importance is not None:
        out = (Path(__file__).resolve().parents[1] / "models" /
               "feature_importance.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        (pd.DataFrame({"feature": feature_names, "importance": importance})
         .sort_values("importance", ascending=False)
         .to_csv(out, index=False))
        log.info("Wrote feature importances to %s", out)


def _native_importance(estimator, n_features):
    """Dig the importances out of whatever model won."""
    model = estimator
    if isinstance(estimator, Pipeline):
        model = estimator.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        return np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    if isinstance(estimator, MultiOutputRegressor):
        per = [est.feature_importances_ for est in estimator.estimators_
               if hasattr(est, "feature_importances_")]
        if per:
            return np.mean(per, axis=0)
    return None


def main():
    log = setup_logging()

    df = feature_store.read_features().dropna(subset=TARGET_COLS)
    feature_names = [c for c in df.columns if c not in NON_FEATURE_COLS]
    log.info("Loaded %d rows with %d features.", len(df), len(feature_names))

    train_df, test_df = time_split(df)
    X_train, y_train = train_df[feature_names], train_df[TARGET_COLS]
    X_test, y_test = test_df[feature_names], test_df[TARGET_COLS]
    log.info("Train rows: %d, test rows: %d", len(train_df), len(test_df))

    results = {}
    fitted = {}
    for name, model in candidate_models().items():
        log.info("Training %s...", name)
        model.fit(X_train, y_train)
        scores = evaluate(y_test, model.predict(X_test))
        results[name] = scores
        fitted[name] = model
        log.info("  %s -> RMSE %.2f  MAE %.2f  R2 %.3f",
                 name, scores["rmse"], scores["mae"], scores["r2"])

    best_name = min(results, key=lambda n: results[n]["rmse"])
    best_scores = results[best_name]
    log.info("Best model: %s (RMSE %.2f, MAE %.2f, R2 %.3f)",
             best_name, best_scores["rmse"], best_scores["mae"],
             best_scores["r2"])

    explain(fitted[best_name], X_train, X_test, feature_names, log)

    flat_metrics = {
        "rmse": best_scores["rmse"],
        "mae": best_scores["mae"],
        "r2": best_scores["r2"],
    }
    registry.save_model(fitted[best_name], feature_names, flat_metrics, best_name)
    log.info("Done.")


if __name__ == "__main__":
    main()
