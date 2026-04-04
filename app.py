import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
import folium
from streamlit_folium import st_folium
from fpdf import FPDF

# -------------------- CONFIG --------------------
API_KEY = "a68ba80d79d7066278b76ec88a96eaae"
st.set_page_config(page_title="Air Quality System", layout="wide")

# -------------------- SESSION STATE --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- UI --------------------
st.title("🌍 Air Quality Monitoring & Prediction System")

# Sidebar for multi-city input
cities_input = st.sidebar.text_area(
    "Enter City Names (comma separated)", "London"
)
cities = [c.strip() for c in cities_input.split(",") if c.strip()]

# Tabs for organization
tab1, tab2, tab3 = st.tabs(["Current AQI", "Prediction", "History & Download"])

# -------------------- FUNCTIONS --------------------
def get_geo(city):
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
    geo_data = requests.get(geo_url).json()
    return geo_data[0]["lat"], geo_data[0]["lon"]

def get_weather(city):
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    w = requests.get(weather_url).json()
    return w["main"]["temp"], w["main"]["humidity"], w["main"]["pressure"]

def get_aqi(lat, lon):
    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    data = requests.get(aqi_url).json()
    return data["list"][0]["main"]["aqi"], data["list"][0]["components"]

def aqi_status(aqi):
    if aqi == 1: return "Good 😊", "green", "Air quality is excellent."
    elif aqi == 2: return "Fair 🙂", "yellow", "Air quality is acceptable."
    elif aqi == 3: return "Moderate 😐", "orange", "Sensitive groups should be cautious."
    elif aqi == 4: return "Poor 😷", "red", "Air quality is unhealthy."
    else: return "Very Poor ☠️", "darkred", "Avoid outdoor activities!"

def train_ml_model():
    # Dummy ML model: In practice, replace with historical AQI dataset
    X = np.array([[10, 20, 200], [20, 30, 300], [30, 40, 400], [40, 50, 500]])
    y = np.array([50, 80, 120, 150])
    model = LinearRegression()
    model.fit(X, y)
    return model

# -------------------- TAB 1: CURRENT AQI --------------------
with tab1:
    st.header("🌫 Current Air Quality")

    map_df = pd.DataFrame(columns=["City", "lat", "lon", "AQI", "Color"])

    for city in cities:
        try:
            lat, lon = get_geo(city)
            temp, humidity, pressure = get_weather(city)
            real_aqi, components = get_aqi(lat, lon)
            status, color, health_msg = aqi_status(real_aqi)

            # Display weather & AQI
            st.subheader(f"{city}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🌡 Temp (°C)", temp)
            col2.metric("💧 Humidity (%)", humidity)
            col3.metric("⚖ Pressure", pressure)
            col4.metric("🌫 AQI", real_aqi)

            st.markdown(f"**Status:** <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
            st.info(f"Health Advice: {health_msg}")

            # Pollutant graph
            df = pd.DataFrame(components.items(), columns=["Pollutant", "Value"])
            fig, ax = plt.subplots()
            ax.bar(df["Pollutant"], df["Value"], color='orange')
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Save history
            record = {
                "City": city,
                "Temp": temp,
                "Humidity": humidity,
                "Pressure": pressure,
                "AQI": real_aqi,
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.history.append(record)

            # Map marker
            map_df = pd.concat([map_df, pd.DataFrame({
                "City": [city], "lat": [lat], "lon": [lon], "AQI": [real_aqi], "Color": [color]
            })])

        except:
            st.error(f"❌ Error fetching data for {city}")

    # Show map
    if not map_df.empty:
        m = folium.Map(location=[map_df["lat"].mean(), map_df["lon"].mean()], zoom_start=2)
        for i, row in map_df.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=10,
                color=row["Color"],
                fill=True,
                fill_opacity=0.7,
                popup=f"{row['City']} AQI: {row['AQI']}"
            ).add_to(m)
        st_folium(m, width=700, height=400)

# -------------------- TAB 2: PREDICTION --------------------
with tab2:
    st.header("🤖 AQI Prediction (Next Day Simulation)")
    model = train_ml_model()

    pred_df = pd.DataFrame(columns=["City", "Predicted AQI"])
    for city in cities:
        try:
            lat, lon = get_geo(city)
            _, components = get_aqi(lat, lon)
            input_data = np.array([[components['pm2_5'], components['pm10'], components['co']]])
            predicted = model.predict(input_data)[0]
            pred_df = pd.concat([pred_df, pd.DataFrame({"City": [city], "Predicted AQI": [round(predicted,2)]})])
        except:
            continue

    st.dataframe(pred_df)

    # Prediction line chart
    st.line_chart(pred_df["Predicted AQI"])

# -------------------- TAB 3: HISTORY & DOWNLOAD --------------------
with tab3:
    st.header("📜 History & Reports")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)

        # Download CSV
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download CSV", csv, "air_quality_report.csv", "text/csv")

        # Download PDF
        if st.button("⬇ Download PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Air Quality Report", 0, 1, "C")
            pdf.set_font("Arial", "", 12)
            for _, row in history_df.iterrows():
                pdf.cell(0, 8, f"{row['Time']} - {row['City']} | Temp: {row['Temp']}°C | Humidity: {row['Humidity']}% | AQI: {row['AQI']}", 0, 1)
            pdf.output("air_quality_report.pdf")
            st.success("PDF Generated! Check your downloads folder.")
            
