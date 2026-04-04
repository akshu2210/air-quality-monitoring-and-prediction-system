import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

# API KEY
API_KEY = "a68ba80d79d7066278b76ec88a96eaae"

st.set_page_config(page_title="Air Quality System", layout="centered")

st.title("🌍 Air Quality Monitoring and Prediction System")

city = st.text_input("Enter City Name", "London")

# Store history
if "history" not in st.session_state:
    st.session_state.history = []

if city:
    try:
        # ---------------- WEATHER DATA ----------------
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        weather_response = requests.get(weather_url).json()

        temp = weather_response["main"]["temp"]
        humidity = weather_response["main"]["humidity"]
        pressure = weather_response["main"]["pressure"]

        st.subheader(f"🌤 Weather Data for {city}")
        col1, col2, col3 = st.columns(3)
        col1.metric("🌡 Temp (°C)", temp)
        col2.metric("💧 Humidity (%)", humidity)
        col3.metric("ضغط Pressure", pressure)

        # ---------------- GEO LOCATION ----------------
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
        geo_data = requests.get(geo_url).json()

        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']

        # ---------------- AQI DATA ----------------
        aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        aqi_data = requests.get(aqi_url).json()

        real_aqi = aqi_data["list"][0]["main"]["aqi"]
        components = aqi_data["list"][0]["components"]

        st.subheader("🌫 Real Air Quality Data")
        st.write("AQI Level:", real_aqi)

        df = pd.DataFrame(components.items(), columns=["Pollutant", "Value"])
        st.dataframe(df)

        # AQI CATEGORY
        def get_aqi_status(aqi):
            if aqi == 1:
                return "Good 😊"
            elif aqi == 2:
                return "Fair 🙂"
            elif aqi == 3:
                return "Moderate 😐"
            elif aqi == 4:
                return "Poor 😷"
            else:
                return "Very Poor ☠️"

        status = get_aqi_status(real_aqi)
        st.success(f"Status: {status}")

        # ---------------- ALERT SYSTEM ----------------
        if real_aqi >= 4:
            st.error("⚠️ Warning: Air quality is unhealthy!")
        elif real_aqi == 3:
            st.warning("⚠️ Moderate air quality. Be cautious.")

        # ---------------- ML PREDICTION ----------------
        X = np.array([
            [components['pm2_5'], components['pm10'], components['co']],
            [10, 20, 200],
            [20, 30, 300],
            [30, 40, 400]
        ])

        y = np.array([50, 80, 120, 150])

        model = LinearRegression()
        model.fit(X, y)

        input_data = np.array([[components['pm2_5'], components['pm10'], components['co']]])
        predicted_aqi = model.predict(input_data)[0]

        st.subheader("🤖 ML Predicted AQI")
        st.write(round(predicted_aqi, 2))

        # ---------------- GRAPH ----------------
        st.subheader("📈 Pollution Graph")

        fig, ax = plt.subplots()
        ax.bar(df["Pollutant"], df["Value"], color='orange')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # ---------------- LINE CHART ----------------
        st.subheader("📊 Trend Simulation")

        trend_data = pd.DataFrame({
            "AQI": np.random.randint(50, 150, size=10)
        })

        st.line_chart(trend_data)

        # ---------------- MAP ----------------
        st.subheader("📍 Location Map")

        map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
        st.map(map_data)

        # ---------------- SAVE HISTORY ----------------
        record = {
            "City": city,
            "AQI": real_aqi,
            "Predicted": round(predicted_aqi, 2),
            "Time": datetime.now().strftime("%H:%M:%S")
        }

        st.session_state.history.append(record)

        # ---------------- HISTORY TABLE ----------------
        st.subheader("📜 History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)

        # ---------------- DOWNLOAD REPORT ----------------
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇ Download Report",
            data=csv,
            file_name='air_quality_report.csv',
            mime='text/csv'
        )

    except:
        st.error("❌ Error fetching data. Check city name or API key.")