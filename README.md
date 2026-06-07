# Pearls AQI Predictor

A small end-to-end machine learning system that predicts the Air Quality Index for Lahore three days into the future. It fetches pollutant and weather data on a schedule, builds features, trains and compares a few models, and registers the best one, then serves the forecast through an API and a dashboard. The whole setup runs on free tiers: Open-Meteo for the data, Hopsworks for the feature store and model registry, and GitHub Actions for automation.

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

- **Feature pipeline** (`feature_pipeline/build_features.py`) loads the most recent data, summarizes it to daily rows, constructs time-based and derived features and the next-3-days targets, and stores them in the feature store.
- **Training pipeline** (`training/train.py`) loads the features, trains and races Ridge, ElasticNet, Random Forest and XGBoost, and registers the best one, but only when it passes a baseline guardrail, so a bad day's model cannot take the place of a good one.
- **Inference API** (`inference_api/app.py`) loads the registered model and returns the forecast as JSON.
- **Dashboard** (`dashboard/app.py`) shows the current AQI, the 3-day outlook, the recent trend, what's driving the prediction, and hazard alerts.

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

You need **Python 3.11 or 3.12**. The Hopsworks client does not currently support 3.13+, so this is worth doing right.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Hopsworks

1. Sign up for a free account at [hopsworks.ai](https://app.hopsworks.ai) and create a project.
2. Create an API key under **Account Settings → API Keys** with the feature store, project, job and model registry scopes.
3. Copy `.env.example` to `.env` and fill it in:

   ```
   HOPSWORKS_API_KEY=your_key_here
   HOPSWORKS_PROJECT=your_project_name
   ```

In case you forget to do this the pipelines will continue to run, but will revert to a local CSV at data/ rather than Hopsworks, which is useful to test things out locally.

> **Windows note:** the Hopsworks client writes a few certificate files to `/tmp`. If you hit a "path not found" error, create a `C:\tmp` folder once and you're set.

## Running it

Populate the feature store with history of about 90 days (only once):

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

## Deploying the dashboard

The dashboard runs on **Streamlit Community Cloud**, which is free and redeploys
itself on every push. To deploy your own copy:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Create a new app pointing at this repo, the `main` branch, and
   `dashboard/app.py` as the main file.
3. Under Advanced settings, add your Hopsworks secrets in TOML format (note the
   quotes, this is not the same as the `.env` file):

   ```toml
   HOPSWORKS_API_KEY = "your_key_here"
   HOPSWORKS_PROJECT = "your_project_name"
   ```
4. Click Deploy. The first build takes a few minutes while it installs the
   dependencies; after that the app is live and refreshes whenever you push.

The dashboard reads the model and features straight from Hopsworks, so it does
not need the FastAPI service running to work.

## Automation

There are two GitHub Actions workflows that keep things fresh: the feature pipeline runs every hour, and the training pipeline runs once a day. For them to reach Hopsworks, add two secrets in the repository under **Settings → Secrets and variables → Actions**:

- `HOPSWORKS_API_KEY`
- `HOPSWORKS_PROJECT`

The workflows can also be initiated manually by the Actions tab.

## Logs

All runs are recorded to the console and a time-stamped file in the `logs/` directory: the size of the dataset, the RMSE/MAE/R² of each model, which model was the winner, the decision made by the guardrail and the push to the registry.

These logs are available in a couple of locations:

- **In each GitHub Actions run:** open the Actions tab, select a run and expand the steps to see the entire output. This is the history of all the scheduled runs.
- **As a downloadable artifact:** at the end of every training run, the `logs/` folder is uploaded as a `training-logs` artifact, attached at the bottom.
- **A committed example:** [`logs/sample_training.log`](logs/sample_training.log) is checked in so you can see what a run looks like without scrolling through the Actions tab. (The live `logs/*.log` files are gitignored since they regenerate.)

## A note on the data

The raw pollutant concentrations are calculated to obtain AQI based on the US EPA bands. I operate on hourly readings, instead of the official averaging windows, and therefore consider the index as a truthful approximation, but not the actual regulatory number.
A more detailed description of what was constructed, the model outputs and constraints are in [REPORT.md](REPORT.md).
