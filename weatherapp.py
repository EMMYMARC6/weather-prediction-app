import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ------------------------------
# Load model and label encoder
# ------------------------------
model = pickle.load(open("weather_dataset_model.sav", "rb"))
label_encoder = pickle.load(open("label_encoder.sav", "rb"))

# ------------------------------
# App Title
# ------------------------------
st.title("🌦 Weather Prediction App - RP Kigali College")
st.write("Predict weather conditions using a trained Decision Tree model.")

# ------------------------------
# Sidebar - Single Input
# ------------------------------
st.sidebar.header("Single Prediction Input")

precipitation = st.sidebar.number_input("Precipitation (mm)", min_value=0.0, max_value=200.0, value=10.0)
temp_max = st.sidebar.number_input("Max Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
temp_min = st.sidebar.number_input("Min Temperature (°C)", min_value=-10.0, max_value=50.0, value=15.0)
wind = st.sidebar.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=5.0)
if st.sidebar.button("Predict Weather"):
    input_features = np.array([[precipitation, temp_max, temp_min, wind]])
    prediction = model.predict(input_features)
    decoded_prediction = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🌍 Predicted Weather: **{decoded_prediction}**")

# ------------------------------
# Batch Prediction from CSV
# ------------------------------
st.header("Batch Prediction from CSV")
st.write("Upload a CSV file with columns: precipitation, temp_max, temp_min, wind")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ['precipitation', 'temp_max', 'temp_min', 'wind']
        if all(col in df.columns for col in required_cols):
            predictions = model.predict(df[required_cols])
            df['Predicted_Weather'] = label_encoder.inverse_transform(predictions)
            st.write(df)
        else:
            st.error(f"CSV must contain columns: {required_cols}")
    except Exception as e:
         st.error(f"Error reading file: {e}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.write("✅ Built by Emmanuel Nshimiyimana | Weather Prediction Project")
