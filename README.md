# Pearls AQI Predictor

A small end-to-end machine learning system that forecasts the Air Quality Index
for **Lahore** three days ahead. It pulls pollutant and weather data on a
schedule, engineers features, trains and compares a handful of models, registers
the best one, and serves the forecast through an API and a dashboard. The whole
thing runs on free tiers — Open-Meteo for data, Hopsworks for the feature store
and model registry, and GitHub Actions for automation.

## How it fits together

```
Open-Meteo API ──► feature pipeline ──► Hopsworks feature store
                                              │
                                              ▼
                                       training pipeline ──► Hopsworks model registry
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼                                        ▼
                   FastAPI service                          Streamlit dashboard
```

- **Feature pipeline** (`feature_pipeline/build_features.py`) fetches the latest
  data, collapses it to daily rows, builds time-based and derived features plus
  the next-3-days targets, and writes them to the feature store.
- **Training pipeline** (`training/train.py`) reads the features, races Ridge,
  ElasticNet, Random Forest and XGBoost against each other, and registers
  whichever has the lowest average RMSE.
- **Inference API** (`inference_api/app.py`) loads the registered model and
  serves the forecast as JSON.
- **Dashboard** (`dashboard/app.py`) shows the current AQI, the 3-day outlook,
  the recent trend, what's driving the prediction, and hazard alerts.

## Project layout

```
data_pipeline/      raw data fetch + EPA AQI calculation
feature_pipeline/   daily feature engineering + backfill
training/           model comparison, SHAP, model registry
inference_api/      FastAPI service
dashboard/          Streamlit app
utils/              config, feature store, registry and shared helpers
notebooks/          exploratory data analysis
.github/workflows/  hourly feature + daily training automation
```

## Getting set up

You need **Python 3.11 or 3.12** — the Hopsworks client doesn't build on 3.13+
yet, so this is worth getting right.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Hopsworks

1. Make a free account at [hopsworks.ai](https://app.hopsworks.ai) and create a
   project.
2. Generate an API key under **Account Settings → API Keys** with the feature
   store, project, job and model registry scopes.
3. Copy `.env.example` to `.env` and fill it in:

   ```
   HOPSWORKS_API_KEY=your_key_here
   HOPSWORKS_PROJECT=your_project_name
   ```

If you skip this step the pipelines still run — they just fall back to a local
CSV under `data/` instead of Hopsworks, which is handy for trying things out
offline.

> **Windows note:** the Hopsworks client writes some certificate files to
> `/tmp`. If you hit a "path not found" error, create a `C:\tmp` folder once and
> you're good.

## Running it

Seed the feature store with ~90 days of history (do this once):

```bash
python feature_pipeline/build_features.py --backfill
```

Then the normal cycle:

```bash
python feature_pipeline/build_features.py     # top up with recent data
python training/train.py                       # train + register the best model
uvicorn inference_api.app:app --reload         # serve the forecast at :8000
streamlit run dashboard/app.py                 # open the dashboard
```

The API gives you `/predict` for the full forecast and `/alert` for just the
hazardous-day summary.

## Automation

Two GitHub Actions workflows keep things fresh: the feature pipeline runs every
hour and the training pipeline runs once a day. For them to reach Hopsworks, add
two repository secrets under **Settings → Secrets and variables → Actions**:

- `HOPSWORKS_API_KEY`
- `HOPSWORKS_PROJECT`

Both workflows can also be triggered by hand from the Actions tab.

## A note on the data

AQI is computed from the raw pollutant concentrations using the US EPA bands. We
work off hourly readings rather than the official averaging windows, so treat
the index as a faithful approximation rather than the exact regulatory figure.
There's a fuller write-up of what was built, the model results and the
limitations in [REPORT.md](REPORT.md).
