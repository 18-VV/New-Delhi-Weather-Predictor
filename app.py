import streamlit as st
import requests
import numpy as np
import datetime

st.set_page_config(page_title="Delhi Weather ML", page_icon="☀️")

# Inject fixed CSS targeting only input text boxes, keeping buttons visible
page_bg_css = """
<style>
/* Sleek dark blue gradient background */
.stApp { 
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
}

/* Force main text, titles, and labels to be white */
.stMarkdown, .stText, p, h1, h2, h3, label { 
    color: white !important; 
}

/* Translucent dark navy sidebar */
[data-testid="stSidebar"] { 
    background-color: rgba(15, 25, 45, 0.85) !important; 
}

/* FIX: Target ONLY text/number input boxes to have dark text */
div[data-baseweb="input"] input {
    color: black !important;
}

/* FIX: Ensure buttons stand out with bold white text and a clean background */
.stButton>button {
    color: white !important;
    background-color: #1e3c72 !important;
    border: 1px solid #ffffff33 !important;
    font-weight: bold !important;
}
.stButton>button:hover {
    background-color: #2a5298 !important;
    border: 1px solid #ffffff66 !important;
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

st.title("☀️ New Delhi Weather Predictor")
st.sidebar.header("Input Current Weather")

st.sidebar.markdown("### 📅 Calendar Override")
selected_date = st.sidebar.date_input("Today's date:", datetime.date.today())

humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 45.0)
pressure = st.sidebar.slider("Pressure (hPa)", 950.0, 1050.0, 1005.0)

day_of_year = selected_date.timetuple().tm_yday
day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
day_cos = np.cos(2 * np.pi * day_of_year / 365.25)

st.sidebar.markdown("### Past 3 Days (Temp °C)")
t1 = st.sidebar.number_input("Yesterday Temp", value=38.0)
t2 = st.sidebar.number_input("2 Days Ago Temp", value=39.0)
t3 = st.sidebar.number_input("3 Days Ago Temp", value=37.0)

st.sidebar.markdown("### Past 3 Days (Humidity %)")
h1 = st.sidebar.number_input("Yesterday Humidity", value=45.0)
h2 = st.sidebar.number_input("2 Days Ago Humidity", value=40.0)
h3 = st.sidebar.number_input("3 Days Ago Humidity", value=42.0)

if st.button("Run Prediction", use_container_width=True):
    payload = {
        "Humidity": humidity, "Pressure": pressure, "Day_Sin": day_sin, "Day_Cos": day_cos,
        "Temp_Lag_1": t1, "Temp_Lag_2": t2, "Temp_Lag_3": t3,
        "Hum_Lag_1": h1, "Hum_Lag_2": h2, "Hum_Lag_3": h3
    }
    
    try:
        res = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if res.status_code == 200:
            st.success(f"🌡️ Looks like tomorrow in New Delhi will be around **{res.json()['predicted_temp']} °C**")
        else:
            st.error("API returned an error.")
    except Exception:
        st.error("Couldn't connect to backend. Is the FastAPI server running?")