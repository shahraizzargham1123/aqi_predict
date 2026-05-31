"""
Streamlit dashboard for the AQI forecast.

It shows where the air sits right now, what the model thinks the next three days
look like, the recent trend, what's driving the prediction, and a clear warning
when any day crosses into hazardous territory.

Run it with:
    streamlit run dashboard/app.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

# On Streamlit Community Cloud the Hopsworks credentials come in as Streamlit
# secrets rather than a .env file. Copy them into the environment before we load
# anything that reads config, so the same code works deployed and locally.
# Accessing st.secrets with no secrets file raises, hence the guard.
try:
    for _key in ("HOPSWORKS_API_KEY", "HOPSWORKS_PROJECT"):
        if not os.getenv(_key) and _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass

from utils import config
from utils.predict import latest_forecast, recent_history

# EPA colour bands, so the numbers feel like the AQI people are used to seeing.
AQI_COLORS = [
    (50, "#00e400", "Good"),
    (100, "#ffde33", "Moderate"),
    (150, "#ff9933", "Unhealthy for Sensitive Groups"),
    (200, "#cc0033", "Unhealthy"),
    (300, "#660099", "Very Unhealthy"),
    (10_000, "#7e0023", "Hazardous"),
]

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def color_for(aqi):
    for ceiling, color, _ in AQI_COLORS:
        if aqi <= ceiling:
            return color
    return AQI_COLORS[-1][1]


@st.cache_data(ttl=1800)
def get_forecast():
    return latest_forecast()


@st.cache_data(ttl=1800)
def get_history(days):
    return recent_history(days)


@st.cache_data(ttl=1800)
def get_importance():
    path = MODELS_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def aqi_metric_card(label, aqi, category):
    """A coloured block showing one AQI number."""
    color = color_for(aqi)
    st.markdown(
        f"""
        <div style="background:{color};padding:18px;border-radius:12px;
                    text-align:center;color:#111;">
            <div style="font-size:15px;font-weight:600;">{label}</div>
            <div style="font-size:42px;font-weight:800;line-height:1.1;">{aqi}</div>
            <div style="font-size:13px;">{category}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️",
                       layout="wide")
    st.title("🌫️ Pearls AQI Predictor")
    st.caption(f"3-day air quality forecast for {config.CITY_NAME}")

    try:
        data = get_forecast()
    except FileNotFoundError:
        st.error("No model or features found yet. Run the feature and training "
                 "pipelines first.")
        st.stop()

    # The headline alert. Loud and red when something's wrong, calm otherwise.
    if data["any_hazardous"]:
        bad_days = ", ".join(d["date"] for d in data["forecast"] if d["hazardous"])
        st.error(f"⚠️ Hazardous air expected on: {bad_days}. Limit outdoor "
                 f"activity and consider a mask.")
    else:
        st.success("✅ No hazardous AQI days in the forecast window.")

    st.subheader("Right now")
    aqi_metric_card("Latest observed AQI", data["current_aqi"],
                    data["current_category"])
    st.caption(f"Based on data up to {data['based_on']} · "
               f"model in use: {data['model']}")

    st.subheader("Next 3 days")
    cols = st.columns(len(data["forecast"]))
    for col, day in zip(cols, data["forecast"]):
        with col:
            aqi_metric_card(f"Day +{day['horizon']} ({day['date']})",
                            day["aqi"], day["category"])

    st.subheader("Recent trend and forecast")
    history = get_history(30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["date"], y=history["aqi"], mode="lines+markers",
        name="Observed", line=dict(color="#1f77b4"),
    ))
    # Stitch the forecast onto the end of the observed line.
    forecast_x = [pd.to_datetime(data["based_on"])] + \
                 [pd.to_datetime(d["date"]) for d in data["forecast"]]
    forecast_y = [data["current_aqi"]] + [d["aqi"] for d in data["forecast"]]
    fig.add_trace(go.Scatter(
        x=forecast_x, y=forecast_y, mode="lines+markers",
        name="Forecast", line=dict(color="#d62728", dash="dash"),
    ))
    fig.add_hline(y=config.HAZARDOUS_AQI_THRESHOLD, line_dash="dot",
                  line_color="#7e0023",
                  annotation_text="Hazardous threshold")
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="AQI",
                      margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    importance = get_importance()
    if importance is not None:
        st.subheader("What's driving the forecast")
        st.caption("Top feature importances (SHAP where available) for the "
                   "day-ahead prediction.")
        top = importance.head(12).iloc[::-1]
        bar = go.Figure(go.Bar(x=top["importance"], y=top["feature"],
                               orientation="h", marker_color="#2c7fb8"))
        bar.update_layout(height=420, margin=dict(t=10),
                          xaxis_title="Importance")
        st.plotly_chart(bar, use_container_width=True)

    with st.expander("Model performance (held-out test set)"):
        metrics = data.get("metrics", {})
        if metrics:
            m = st.columns(3)
            m[0].metric("RMSE", f"{metrics.get('rmse', float('nan')):.2f}")
            m[1].metric("MAE", f"{metrics.get('mae', float('nan')):.2f}")
            m[2].metric("R²", f"{metrics.get('r2', float('nan')):.3f}")
        st.caption("Averaged across the three forecast horizons.")


if __name__ == "__main__":
    main()
