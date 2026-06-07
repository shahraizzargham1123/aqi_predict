"""
Training pipeline: pull features from the store, race a few models against each
other, keep the best one and register it.

I compare Ridge, ElasticNet, Random Forest and XGBoost. Each one predicts all
three horizons at once (day+1, +2, +3) so a single model gives the whole 3-day
outlook. Each model's hyperparameters are tuned with a time-series cross
validation (a grid search over walk-forward splits) before they're judged on a
held-out test set, and the winner is whichever has the lowest average RMSE
across the three days. There's room left to slot in a TensorFlow/LSTM model
later, but the brief for now is classical models, so that's what I ship.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
    """The line-up, each paired with a grid of settings to search over. Rather
    than hardcode hyperparameters I let cross-validation pick them, which keeps
    the linear models from over-extrapolating and gives the trees a fair shot.

    Linear models sit behind a scaler; the tree models don't need one. The grids
    are kept deliberately small because there isn't much data to validate on.
    XGBoost only joins if it's installed."""
    models = {
        "ridge": (
            Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]},
        ),
        "elasticnet": (
            Pipeline([("scale", StandardScaler()),
                      ("model", ElasticNet(max_iter=10000))]),
            {"model__alpha": [0.05, 0.1, 0.5, 1.0, 5.0],
             "model__l1_ratio": [0.2, 0.5, 0.8]},
        ),
        "random_forest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {"n_estimators": [200, 400],
             "max_depth": [4, 8, 12],
             "min_samples_leaf": [1, 2]},
        ),
    }
    if HAS_XGBOOST:
        # XGBoost isn't natively multi-output, so wrap one regressor per day.
        # The grid params need the "estimator__" prefix to reach through the
        # MultiOutputRegressor wrapper.
        models["xgboost"] = (
            MultiOutputRegressor(XGBRegressor(
                random_state=42, subsample=0.9, colsample_bytree=0.9,
            )),
            {"estimator__n_estimators": [200, 400],
             "estimator__max_depth": [2, 3, 4],
             "estimator__learning_rate": [0.05, 0.1]},
        )
    return models


def evaluate(y_true, y_pred):
    """RMSE, MAE and R², averaged over the three horizons but also kept per-day
    so I can see which days are harder to call."""
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
    """Time series, so no shuffling. The last chunk of days is the test set I
    judge the candidate models on."""
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * (1 - test_fraction))
    return df.iloc[:split], df.iloc[split:]


def baseline_rmse(test_df, y_test):
    """RMSE of the naive persistence forecast: assume the next three days just
    look like today. It's the bar any model I deploy should clear, and it's
    recomputed each run so a calm or stormy week is judged on its own terms."""
    today = test_df["aqi"].to_numpy().reshape(-1, 1)
    persistence = np.repeat(today, len(TARGET_COLS), axis=1)
    return evaluate(y_test, persistence)["rmse"]


def explain(best_estimator, X_train, X_test, feature_names, log):
    """Save feature importances. I use SHAP when it's available, and always
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

    # Walk-forward splits for the hyperparameter search, so tuning never peeks at
    # the future to tune the present.
    cv = TimeSeriesSplit(n_splits=4)

    results = {}
    fitted = {}
    for name, (estimator, grid) in candidate_models().items():
        log.info("Tuning %s...", name)
        search = GridSearchCV(
            estimator, grid, cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        scores = evaluate(y_test, best.predict(X_test))
        results[name] = scores
        fitted[name] = best
        log.info("  %s -> RMSE %.2f  MAE %.2f  R2 %.3f  | best: %s",
                 name, scores["rmse"], scores["mae"], scores["r2"],
                 search.best_params_)

    best_name = min(results, key=lambda n: results[n]["rmse"])
    best_scores = results[best_name]
    log.info("Best candidate: %s (RMSE %.2f, MAE %.2f, R2 %.3f)",
             best_name, best_scores["rmse"], best_scores["mae"],
             best_scores["r2"])

    # Guardrail: only promote the new model if it can beat the naive "tomorrow
    # looks like today" baseline (within tolerance). A retrain that can't even
    # do that is a bad model, so I keep whatever's already serving instead.
    baseline = baseline_rmse(test_df, y_test)
    ceiling = baseline * (1 + config.PROMOTION_TOLERANCE)
    log.info("Persistence baseline RMSE %.2f (promotion ceiling %.2f).",
             baseline, ceiling)
    if best_scores["rmse"] > ceiling:
        log.warning(
            "Keeping current model: best candidate %s scores RMSE %.2f, which "
            "can't beat the persistence baseline within tolerance. Not promoting.",
            best_name, best_scores["rmse"])
        log.info("Done.")
        return

    # Cleared to ship. Refit the winner on every row I have, the recent days
    # included, so the deployed model has learned from all the data rather than
    # just the training slice.
    X_all, y_all = df[feature_names], df[TARGET_COLS]
    final_model = clone(fitted[best_name])
    final_model.fit(X_all, y_all)
    log.info("Refit %s on all %d rows for deployment.", best_name, len(df))

    explain(final_model, X_train, X_test, feature_names, log)

    flat_metrics = {
        "rmse": best_scores["rmse"],
        "mae": best_scores["mae"],
        "r2": best_scores["r2"],
    }
    registry.save_model(final_model, feature_names, flat_metrics, best_name)
    log.info("Done.")


if __name__ == "__main__":
    main()
