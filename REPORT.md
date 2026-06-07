# Pearls AQI Predictor Report

## What I set out to build

I set out to build an end-to-end machine learning system to predict the Air Quality Index in Lahore three days in advance, running on free infrastructure. Not a model in a notebook, but the entire process: data is automatically collected, features are engineered and stored, models are trained and compared and registered, the best one is served via an API and a dashboard, and the entire process is kept up-to-date on a schedule without me having to touch it.

The rest is all constructed and in operation. The data is updated hourly, the model is retrained daily and the dashboard displays real-time 3-day forecast and hazard alerts. This report takes a step-by-step tour of its assembly and, not the least important, the issues I encountered on the way to it and how I managed to resolve them, since that is where the bulk of the real work was done.

## The stack, and why

| Concern | Choice | Why |
| ------- | ------ | --- |
| Raw data | Open-Meteo Air Quality + Weather | Free, no API key, and provides both live and historical data, which the backfill requires |
| Feature store + model registry | Hopsworks | Generous free tier with the feature store and model registry in one place |
| Models | scikit-learn + XGBoost | The linear and tree-ensemble families are both well represented; the brief focuses on classical models |
| Inference API | FastAPI | Light, typed, fast to stand up |
| Dashboard | Streamlit | Fastest path to an interactive view |
| Automation | GitHub Actions | Free scheduled runs, no dedicated orchestrator to run |

One of the design choices that proved to be a success several times: I placed a thin abstraction in front of the feature store and the model registry in such a way that both of them will resort to local files when not using Hopsworks. That allowed me to build offline and it is a backup in case the managed service has a bad day.

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

## Data collection and the AQI calculation

Open-Meteo provides me with hourly pollutant concentrations (PM2.5, PM10, ozone, NO₂, SO₂, CO) and weather (temperature, humidity, wind, surface pressure). The raw concentrations are not the AQI people recognise, so I transform them using the US EPA breakpoint tables: each pollutant gets a sub-index, and the overall AQI is the worst of them. The gases are first converted from µg/m³ to ppb/ppm, since the EPA bands are measured in those units.

One candid reservation I wish to make: the official AQI averages across windows (24h particulates, 8h ozone and CO), but I calculate the sub-index by just taking the hourly values. That is a typical simplification for a forecasting feature, but it means my figure is a faithful approximation, not the exact regulatory figure.

## Feature engineering

Since the prediction is on a daily basis, I roll the hourly data up to one row per day and create the features on top:

- **Time features:** day, month, day-of-week, day-of-year, a weekend flag, and cyclical sine/cosine encodings of month and day-of-week so the model can view the calendar as a wrap-around (December goes back to January).
- **Derived features:** change and change rate of day-over-day AQI, 1-3 day AQI lags, rolling 3- and 7-day means, a rolling 3-day standard deviation, and two weather lags. These are the ones that bear the burden.
- **Targets:** AQI one, two and three days ahead, so a single model generates the entire 3-day forecast simultaneously.

The feature store is seeded with a 90-day backfill and kept topped up by the hourly job. The common data issues are also addressed by the pipeline despite Open-Meteo being clean: it inner-joins pollutants and weather by the timestamp, removes rows it can't score, averages by hour to create daily rows, and deduplicates by a (city, date) key to avoid duplicates on reruns.

## What the EDA showed

I explored the raw 90 days before modelling (`notebooks/eda.ipynb`). The remainder of the work was influenced by a couple of trends:

- AQI is persistent: the present is a good indicator of the future, which is why the lag and rolling features earn their keep.
- **PM2.5 and PM10** take the lead of the index, which is in line with the fact that particulates are Lahore's primary issue.
- **Wind clears the air**: higher wind speeds correlate with lower AQI.
- There's a clear daily rhythm, which is what the time features are for.

## The models and how I pick one

Each training run is a competition between four models, which predict all three days simultaneously: Ridge, ElasticNet, Random Forest and XGBoost. The linear ones are positioned behind a scaler, the trees do not require one and the split is time based (most recent 20% held out, no shuffling). Hyperparameters of all models are optimized using a time-series cross-validation grid search before they are evaluated, so I am not comparing each model against some arbitrary default. The lowest RMSE on the held-out test set wins.

A recent run on 91 daily rows (72 train / 19 test):

| Model         | RMSE      | MAE       | R²        |
| ------------- | --------- | --------- | --------- |
| **XGBoost**   | **34.21** | **21.60** | **0.130** |
| Random Forest | 34.75     | 24.45     | 0.104     |
| ElasticNet    | 35.90     | 24.33     | 0.040     |
| Ridge         | 36.28     | 23.93     | 0.017     |

The tree ensembles have a tendency to push out the linear models, but it is near and the order changes slightly run to run. To compare, naively assuming that tomorrow is like today has an RMSE of approximately 43 on the same test set, thus all tuned models pass that bar.

There are two things that occur prior to a new model becoming live and both were a direct result of issues that I encountered (explained in the challenges section):

- **A guardrail.** Only a freshly trained model is promoted when it wins over the naive persistence baseline with a small tolerance. In case a retrain of a day comes out less well, it is rejected and the already in service model continues to run.
- **A refit on all the data.** After a model has been trained and cleared the guardrail, it's retrained on the entire dataset (the held-out slice included) and saved, so the deployed model has learned from every row I have.

The dashboard and API are always equipped with the latest model which has passed the guardrail.

## Explaining the predictions

Once a winner has been selected, the pipeline will run SHAP on the day-ahead prediction (with a backup to the importances of the model itself in the event of SHAP failure) and store the output to the dashboard. Thus there is always a simple answer to why this forecast, and it is the recent-AQI lags and rolling means doing most of the talking.

## The web app

The **FastAPI** service has three endpoints: a health check, a predict endpoint to get the entire 3-day forecast (tagged by day with its EPA category and a hazard flag), and an alert endpoint to get only the hazardous days.

The user-friendly interface is the Streamlit dashboard: the present AQI in their EPA colour, three forecast cards, the recent trend with the forecast drawn on and the hazardous threshold marked, the SHAP importance chart, and the metrics of the model. When any of the forecast days exceeds AQI 151, a red banner is displayed. The API and the dashboard share the same shared prediction code, and thus cannot diverge.

The dashboard is deployed on Streamlit Community Cloud, a free host that redeploys it on every push. There the Hopsworks credentials arrive as Streamlit secrets rather than a `.env` file, and the app copies them into the environment on startup so the exact same code runs locally and in the cloud.

## Automation

The schedule is taken care of by two GitHub Actions workflows: the feature pipeline is run every hour and the training pipeline is run once a day, both of which read their Hopsworks credentials out of repository secrets. The training job also posts its logs as a build artifact, therefore, all the runs are auditable.

## Challenges I ran into and how I fixed them

It was here that the bulk of the actual work was. The happy-path code was the simple bit; it was the time that it took to get it to actually run, unattended, on a Windows machine and on free infrastructure.

### 1. Hopsworks wouldn't install on Python 3.14

I was using Python 3.14 on my machine and pip install hopsworks did not work. The import of the now-obsolete (since Python 3.12) imp module is still a dependency of one of its modules. I ensured that it was not a one-off and then developed a special virtual environment of Python 3.11 to work on the project. All install and run there and the GitHub Actions workflows support Python 3.11 as well, making local and CI identical.

### 2. Hopsworks broke on Windows over a `/tmp` path

After installation, even logging in was not possible: the client attempts to write its TLS certificates to the non-existent Windows path of /tmp, and crashed with a path not found error. The login itself was also okay (my API key was valid); it was merely the cert folder. I have made it work in two ways: I pass an explicit cert folder which points to the actual system temp directory when I log in, and I created a C:\tmp directory to allow the few other hardcoded writes to the /tmp directory (the Kafka PEM files). On Linux, where CI is supported, there is no need to create a workaround since the /tmp is built-in, thus this workaround is only Windows-specific and does not cause harm elsewhere.

### 3. Feature inserts needed a Kafka dependency

The first time I tried to write features to Hopsworks, the insert failed asking for confluent-kafka. Hopsworks streams offline inserts via Kafka, which isn't included in the base package. I changed the requirement to `hopsworks[python]==4.7.*`. The `[python]` extra pulls in Kafka, and pinning to 4.7 aligns with the Hopsworks backend version (a mismatch warning was nudging me to 4.7). After that the 90-day backfill landed cleanly.

### 4. An Open-Meteo outage crashed the hourly job

I received a mail that the pipeline of the feature running had failed. At the time I reproduced it, the weather endpoint was responding with a temporary outage on the Open-Meteo side, and had nothing to do with my code. However, my fetcher lacked strength and thus a single blip killed the entire run and created a false alarm. I made two changes: **retry on transient errors with exponential backoff** (502/503/timeouts) to ride out hiccups; and a **graceful skip** in the hourly job, so that in the event of an actual outage of the API it logs the condition and gracefully exits rather than failing. That is safe since a lost hour will not lose anything: the following run will re-pull a 14-day window and make up. Actual errors (bad credentials, a code bug) will never go unnoticed, and I am never oblivious to real issues.

### 5. The 3-day forecast was wildly over-predicting

I saw that the day-3 prediction was at about 200 as the real AQI was at about 150 and even decreasing. To dig deeper, the reason was a mix: the model served was a linear model with hand-picked, under-regularized settings, and the 7-day rolling-average feature was still reflecting a spike in AQI in late-May (~240). A linear model multiplies its coefficients by those uplifted inputs and joyously extrapolates beyond anything that it has observed. I resolved this issue by tuning the regularization of each model using cross-validation rather than guessing. The search preferred robust regularization throughout, which draws the extreme predictions towards sane values. The day-3 figure dropped from 200 to about 146, which was exactly what the official AQI sites were predicting.

### 6. The "best" model never actually updated

This was the most subtle. I was working on the best RMSE model, but the application remained stuck on an initial model. The rationale: each day retrain is tested on a new window, thus, their RMSE scores cannot be compared. One of the early models had been fortunate to be tested on a calm, smooth week and got a low RMSE that later models, tested on more difficult, spikier weeks, were never able to beat. So "best by RMSE" was really "luckiest test week", and it was serving a model I had already shown made poor predictions.

I had to work out a couple of options. My initial attempt was a **frozen validation window**, where I scored each model on the same fixed dates in order to make the numbers comparable. It was effective, but I did not enjoy permanently carving those days out of the data, and a hardcoded window ages awkwardly over time. So I decided on a cleaner design: every daily run honestly picks the best of its four candidates, serves the latest model, and protects that with a guardrail: a new model is not promoted until it beats the naive persistence baseline, otherwise the current model continues to serve. That provides me with a genuine quality gate without feigning incomparable scores as similar. I also ensured that the deployed model is refit on all the data after selection, to ensure that nothing goes to waste.

### 7. Tree models crashed the inference path

During the testing of the switch to XGBoost, prediction went out of control with a dtype error. The issue was with the construction of the single input row. Pandas Series of a single row converted all the columns into object, which scikit-learn linear models silently accept but is an outright rejection by XGBoost. It was a latent bug all along, and was only revealed by the fact that a linear model was being served. I resolved it by constructing the input as a good one-row DataFrame and coercing numeric dtypes, so any type of model is fine.

## Being honest about accuracy

The figures are not spectacular, but are good, and it is well to tell the truth. The best model lands at an R² of about 0.1, thus explaining only a fraction of the daily variation, with an RMSE in the mid-30s on an index that ranges from about 50 to 240. It does surpass the naive persistence baseline (that is the level I set before rolling anything out), but it is a fair description to say that it is modestly better than guessing.

The causes need no secret: the number of training rows is so small (about 70) to make a three-day-ahead forecast, the target is actually volatile, and I am making forecasts based solely on history without conditioning the days ahead on the weather *forecast*. The latter is a conscious decision: giving the model the future weather would make this less of an honest prediction problem, so I left it history-only.

The positive aspect is that the only thing which gets better with time is accuracy.
The hourly job continues to add rows to the training set, so the training set is expanding daily and the identical pipeline ought to refine during the next few weeks without any code modifications.

## What I'd do next

- **Let the history grow.** The same models would be enhanced as the dataset would grow by itself in the next few weeks.
- **Engineer richer history-only features**: longer lags, trend/momentum, pollutant-specific signals.
- **Add a deep-learning model.** The training code already has a clean slot of a TensorFlow/LSTM candidate; as the amount of data increases it is worth adding to the race.
- **Proper time-series cross-validation reporting** to monitor the accuracy whether it is actually improving with the accumulation of data.

## Mapping to the brief

| Requirement                                           | Status                           |
| ----------------------------------------------------- | -------------------------------- |
| Fetch weather + pollutant data from an external API   | Done (Open-Meteo)                |
| Time-based + derived features (incl. AQI change rate) | Done                             |
| Store features in a feature store                     | Done (Hopsworks, local fallback) |
| Historical backfill                                   | Done (90 days)                   |
| Train and compare multiple models                     | Done (4 models, tuned)           |
| Evaluate with RMSE, MAE, R²                           | Done                             |
| Store model in a registry                             | Done (Hopsworks)                 |
| Hourly feature + daily training automation            | Done (GitHub Actions)            |
| Web dashboard with live + forecast AQI                | Done (Streamlit)                 |
| EDA                                                   | Done (notebook)                  |
| SHAP/LIME explainability                              | Done (SHAP)                      |
| Hazardous AQI alerts                                  | Done                             |
