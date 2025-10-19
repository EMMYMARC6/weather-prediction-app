import streamlit as st
import requests
import pickle
import pandas as pd
import altair as alt  # For graphs

# -------------------------------
# Load the trained Decision Tree model and LabelEncoder
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
# API key
# -------------------------------
api_key = "a2193b395f60ad5867e08c4e2d6e64eb"  # Replace with your OpenWeatherMap key

# -------------------------------
# Button to fetch live and forecast data
# -------------------------------
if st.button("Get Live & 5-Day Weather Forecast"):
    
    # ---------- Current weather ----------
    url_current = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response_current = requests.get(url_current)
    
    if response_current.status_code == 200:
        data_current = response_current.json()
        
        temp_max = data_current['main']['temp_max']
        temp_min = data_current['main']['temp_min']
        wind = data_current['wind']['speed']
        precipitation = data_current.get('rain', {}).get('1h', 0.0)

        st.success(f"🌡️ Temperature Max: {temp_max} °C")
        st.info(f"🌡️ Temperature Min: {temp_min} °C")
        st.warning(f"🌧️ Precipitation: {precipitation} mm")
        st.info(f"💨 Wind: {wind} m/s")

        df_current = pd.DataFrame([[precipitation, temp_max, temp_min, wind]],
                                  columns=['precipitation','temp_max', 'temp_min', 'wind'])
        prediction_num = model.predict(df_current)[0]
        prediction_label = le.inverse_transform([prediction_num])[0]
        st.write("### 🌤️ Predicted Current Weather Condition:")
        st.success(prediction_label)

        # Bar chart for current weather
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
        st.error("⚠️ City not found or API error for current weather.")
    
    # ---------- 5-day / 3-hour forecast ----------
    url_forecast = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response_forecast = requests.get(url_forecast)
    
    if response_forecast.status_code == 200:
        data_forecast = response_forecast.json()
        
        times = []
        predictions_labels = []
        
        for entry in data_forecast['list']:
            t_max = entry['main']['temp_max']
            t_min = entry['main']['temp_min']
            wind_speed = entry['wind']['speed']
            precip = entry.get('rain', {}).get('3h', 0.0)
            time = entry['dt_txt']
            
            df_forecast = pd.DataFrame([[precip, t_max, t_min, wind_speed]],
                                       columns=['precipitation','temp_max', 'temp_min', 'wind'])
            pred_num = model.predict(df_forecast)[0]
            pred_label = le.inverse_transform([pred_num])[0]
            
            times.append(time)
            predictions_labels.append(pred_label)
        
        # Display forecast table
        st.subheader("📅 5-Day Weather Predictions (Every 3 hours)")
        forecast_df = pd.DataFrame({'Time': times, 'Predicted Weather': predictions_labels})
        st.dataframe(forecast_df)
        
        # Plot predicted weather trend
        st.subheader("📈 Weather Trend for Next 5 Days")
        chart_trend = alt.Chart(forecast_df).mark_line(point=True).encode(
            x='Time',
            y='Predicted Weather',
            tooltip=['Time','Predicted Weather']
        ).properties(width=800, height=400, title="5-Day Weather Trend")
        st.altair_chart(chart_trend, use_container_width=True)
    
    else:
        st.error("⚠️ API error while fetching 5-day forecast.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by Nshimiyimana Emmanuel © 2025")
