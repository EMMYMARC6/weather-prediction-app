import streamlit as st
import requests
import pickle
import pandas as pd
import altair as alt  # For graphs

# -------------------------------
# Load the trained Decision Tree model
# -------------------------------
model = pickle.load(open('weather_dataset_model.sav', 'rb'))
le = pickle.load(open('label_encoder.sav', 'rb'))  # Make sure you saved the encoder during training

# -------------------------------
# Streamlit App Title and Description
# -------------------------------
st.title("🌤️ Live Weather Prediction App (Rubavu District)")
st.write("This app fetches live weather data online and predicts the weather condition using a trained Decision Tree model.")

# -------------------------------
# Input: City name
# -------------------------------
city = st.text_input("Enter your city name", "Rubavu")

# -------------------------------
# API setup
# -------------------------------
api_key = "a2193b395f60ad5867e08c4e2d6e64eb"  # 🔑 Replace with your real OpenWeatherMap key
url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

# -------------------------------
# Fetch and display live weather data
# -------------------------------
if st.button("Get Live Weather Data"):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()

        # Extract live weather data
        temp_max = data['main']['temp_max']
        temp_min = data['main']['temp_min']
        wind = data['wind']['speed']  # renamed variable
        precipitation = data.get('rain', {}).get('1h', 0.0)

        # Display fetched data
        st.success(f"🌡️ Temperature Max: {temp_max} °C")
        st.info(f"🌡️ Temperature Min: {temp_min} °C")
        st.warning(f"🌧️ Precipitation: {precipitation} mm")
        st.info(f"💨 Wind: {wind} m/s")

        # Prepare data for model prediction
        df = pd.DataFrame([[temp_max, temp_min, precipitation, wind]],
                          columns=['precipitation','temp_max', 'temp_min', 'wind'])
        # Predict numeric label
        prediction_num = model.predict(df)[0]

        # Convert numeric label to original weather string using LabelEncoder
        prediction_label = le.inverse_transform([prediction_num])[0]




        # Predict weather condition
        prediction = model.predict(df)[0]
        st.write("### 🌤️ Predicted Weather Condition:")
        st.success(prediction)

        # -------------------------------
        # Visualization (Bar Chart)
        # -------------------------------
        st.subheader("📈 Live Weather Data Visualization")

        chart_data = pd.DataFrame({
            'Parameter': ['Temp_Max', 'Temp_Min', 'Precipitation', 'Wind'],
            'Value': [temp_max, temp_min, precipitation, wind]
        })

        chart = alt.Chart(chart_data).mark_bar(color='skyblue').encode(
            x='Parameter',
            y='Value'
        ).properties(width=600, height=400, title="Current Weather Parameters")

        st.altair_chart(chart, use_container_width=True)

    else:
        st.error("⚠️ City not found or API error. Please try again.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by Nshimiyimana Emmanuel © 2025")
