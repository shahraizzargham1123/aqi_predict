# Pearls AQI Predictor — Report

## The idea

Predict Lahore's Air Quality Index three days ahead, with everything running on
free tools. Data comes from Open-Meteo, the feature store and model registry live
in Hopsworks, and GitHub Actions keeps it all ticking over. Every part of the
brief is built and working: data collection, feature engineering, a feature
store, model training and comparison, a registry, scheduled automation, an API, a
dashboard, EDA, SHAP explanations and hazard alerts.

The honest headline is that the plumbing is solid but the forecast accuracy is
only okay — and that's a data-volume problem, not a design one. I'll get to it.

## How the data flows

Open-Meteo hands over hourly pollutant and weather readings. The pollutant
concentrations get turned into a real AQI using the US EPA breakpoint tables (each
pollutant gives a sub-index, the worst one wins). One caveat worth being upfront
about: the official AQI averages over 24h/8h windows and I compute it straight
off hourly values, so it's a faithful approximation rather than the exact
regulatory number.

Since the forecast is daily, the hourly data gets rolled up to one row per day.
On top of that sit the features: calendar bits (day, month, day-of-week, plus
cyclical encodings so December wraps back to January), and the derived signals
that actually carry the weight — AQI lags, rolling 3- and 7-day averages, the
day-over-day change rate, and a couple of weather lags. The targets are AQI one,
two and three days out, so one model gives the whole 3-day picture. A 90-day
backfill seeds everything and the hourly job tops it up.

## What the EDA told me

A few patterns stood out and shaped the rest of the work. AQI is sticky — today
is a strong hint about tomorrow, which is why the lag features earn their keep.
PM2.5 and PM10 dominate the index, matching the fact that particulates are
Lahore's usual problem. Wind clears the air. And there's a clear daily rhythm,
which is what the time features are for.

## The models

Four models race each other, each predicting all three days at once: Ridge,
ElasticNet, Random Forest and XGBoost. The linear ones sit behind a scaler, the
trees don't need it, and the split is time-based (most recent 20% held out, no
shuffling). Lowest average RMSE wins.

A typical run on 82 daily rows looked like this:

| Model | RMSE | MAE | R² |
| --- | --- | --- | --- |
| **ElasticNet** | **24.76** | **20.88** | **-0.21** |
| Random Forest | 31.40 | 26.83 | -0.85 |
| XGBoost | 35.33 | 30.18 | -1.82 |
| Ridge | 37.75 | 32.23 | -1.63 |

ElasticNet wins and gets registered. The exact numbers wobble as the backfill
window rolls forward, but the order holds — the regularised linear model keeps
beating the tree ensembles, which is what you'd expect when there isn't much data
to feed a forest. The daily training job keeps whatever's registered fresh, and
that's what the API and dashboard serve.

## Explaining the predictions

Once a winner is picked, the pipeline runs SHAP on the day-ahead prediction (with
a fallback to the model's own importances if SHAP ever chokes) and saves the
result for the dashboard to show. So there's always a straight answer to "why this
forecast" — and it's the recent-AQI lags and rolling means doing most of the
talking.

## Serving it

The FastAPI service has three endpoints: a health check, `/predict` for the full
3-day forecast (each day tagged with its EPA category and a hazard flag), and
`/alert` for just the dangerous days. The Streamlit dashboard wraps it in
something friendlier — current AQI in its EPA colour, three forecast cards, the
recent trend with the forecast drawn on and the hazardous line marked, the SHAP
chart, and the model's test metrics. The moment any day crosses AQI 151, a red
banner shows up.

## Keeping it running

Two GitHub Actions workflows do the automation: features every hour, training once
a day, both pulling their Hopsworks credentials from repository secrets. The
training run also uploads its logs so you can see exactly what happened.

## Being honest about accuracy

The held-out R² sits around zero or just below, which means the model isn't yet
reliably beating a naive "tomorrow looks like today" guess on the test days. The
reasons are no mystery: 90 days is only ~65 training rows, three-day-ahead AQI is
genuinely volatile, and I'm forecasting purely from history without feeding in the
weather *forecast* for the days ahead.

The good news is the easiest fix is just time — the hourly job keeps appending, so
the training set grows on its own and the same models should sharpen up over the
coming weeks. Beyond that, the obvious next moves are bringing in forecast weather
as features, tuning hyperparameters properly, and dropping a TensorFlow/LSTM model
into the slot the training code already leaves open for it.
