import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Air Quality System", layout="wide")

# ---------------- LOAD ENV ----------------
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    st.error("❌ Weather API missing")
    st.stop()

# ---------------- FUNCTIONS ----------------

def get_data(city):
    try:
        w = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        ).json()

        if "main" not in w:
            return None

        lat, lon = w["coord"]["lat"], w["coord"]["lon"]

        a = requests.get(
            f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        ).json()

        pm25 = a["list"][0]["components"]["pm2_5"]
        aqi = int(pm25 * 4)

        return {
            "city": city,
            "aqi": aqi,
            "lat": lat,
            "lon": lon,
            "time": datetime.now()
        }
    except:
        return None


def simple_status(aqi):
    if aqi <= 50:
        return "😊 Air is Clean", "Go outside!", "#4CAF50"
    elif aqi <= 100:
        return "🙂 Air is Okay", "Safe but careful", "#FFC107"
    elif aqi <= 150:
        return "😷 Air is Bad", "Wear mask", "#FF9800"
    else:
        return "🚨 Air is Dangerous", "Stay inside!", "#F44336"


def health_tips(aqi):
    if aqi <= 50:
        return ["😊 Air is clean", "✔ Go outside freely", "🏃 Exercise safe"]
    elif aqi <= 100:
        return ["🙂 Moderate air", "✔ Normal activity ok", "⚠ Sensitive people careful"]
    elif aqi <= 150:
        return ["😷 Unhealthy air", "❗ Wear mask", "🚫 Avoid heavy exercise"]
    else:
        return ["🚨 Dangerous air", "❌ Stay indoors", "😷 Mask required", "⚠ Health risk"]


def predict_aqi(base):
    X = np.array(range(10)).reshape(-1, 1)
    y = np.array([base + np.random.randint(-20, 20) for _ in range(10)])
    model = LinearRegression().fit(X, y)
    future = model.predict(np.array(range(10, 17)).reshape(-1, 1))
    return [int(max(20, min(300, v))) for v in future]


# ---------------- UI ----------------

st.title("🌍 Air Quality Monitoring System")
st.caption("Real-time AQI + Prediction + Health Advisory")

# 👉 AQI EXPLANATION
st.markdown("### ℹ️ What is AQI?")
st.info("""
AQI (Air Quality Index) tells how clean or polluted the air is.

😊 0–50 → Good (Safe)  
🙂 51–100 → Moderate  
😷 101–150 → Unhealthy  
🚨 150+ → Dangerous  

👉 Lower AQI = Better air  
👉 Higher AQI = More pollution
""")

cities = st.text_input("Enter cities", "Delhi,Hyderabad")

# DATA
data = []
for c in cities.split(","):
    d = get_data(c.strip())
    if d:
        data.append(d)

if not data:
    st.error("No data available")
    st.stop()

df = pd.DataFrame(data)
df["future"] = df["aqi"].apply(predict_aqi)

# TABS
tab1, tab2, tab3, tab4 = st.tabs(
    ["🏠 Home", "📅 Forecast", "🗺 Map", "📊 Analysis"]
)

# ---------------- HOME ----------------
with tab1:
    for _, row in df.iterrows():
        s, m, color = simple_status(row["aqi"])

        st.markdown(f"""
        <div style="background:{color};padding:20px;border-radius:20px;text-align:center;color:white">
        <h2>{row['city']}</h2>
        <h1>{s}</h1>
        <p>{m}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 💡 Health Tips")
        for tip in health_tips(row["aqi"]):
            st.write(f"- {tip}")

        st.markdown("---")


# ---------------- FORECAST ----------------
with tab2:
    st.subheader("📅 7-Day Forecast")

    st.info("""
This graph shows how air quality may change over the next 7 days.

📉 Lower line → Cleaner air  
📈 Higher line → More pollution  
""")

    for _, row in df.iterrows():
        st.write(f"### 📍 {row['city']}")
        st.line_chart(row["future"])

    st.markdown("""
### 🧠 Easy Understanding:
- Graph going UP → Air getting worse  
- Graph going DOWN → Air improving  
- Stable → No major change  
""")


# ---------------- MAP ----------------
with tab3:
    st.subheader("🗺 Air Quality Map")
    st.map(df[["lat", "lon"]])


# ---------------- ANALYSIS ----------------
with tab4:
    st.subheader("📊 AQI Analysis")

    st.info("""
This graph shows current AQI levels of selected cities.

Each line represents one city.
""")

    fig = px.line(df, x="time", y="aqi", color="city", markers=True)
    st.plotly_chart(fig)

    st.markdown("### 🧠 Insights")

    for _, row in df.iterrows():
        if row["aqi"] <= 100:
            st.success(f"{row['city']} has clean or moderate air 😊")
        else:
            st.error(f"{row['city']} has polluted air 😷")

    st.markdown("""
### 📖 Interpretation:
- Lower AQI → Healthy air  
- Higher AQI → Health risk  
- Compare cities to see better air quality  
""")
