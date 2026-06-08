from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("weather_model.pkl")

class WeatherInput(BaseModel):
    Humidity: float
    Pressure: float
    Day_Sin: float
    Day_Cos: float
    Temp_Lag_1: float
    Temp_Lag_2: float
    Temp_Lag_3: float
    Hum_Lag_1: float
    Hum_Lag_2: float
    Hum_Lag_3: float

@app.post("/predict")
def predict_weather(input_data: WeatherInput):
    features = np.array([[
        input_data.Humidity, input_data.Pressure, input_data.Day_Sin, input_data.Day_Cos,
        input_data.Temp_Lag_1, input_data.Temp_Lag_2, input_data.Temp_Lag_3,
        input_data.Hum_Lag_1, input_data.Hum_Lag_2, input_data.Hum_Lag_3
    ]])
    prediction = model.predict(features)[0]
    return {"predicted_temp": round(float(prediction), 2)}