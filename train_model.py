import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
import joblib

url = "https://archive-api.open-meteo.com/v1/archive?latitude=28.6139&longitude=77.2090&start_date=2021-01-01&end_date=2023-12-31&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure&timezone=Asia%2FKolkata"
response = requests.get(url)
data = response.json()

df_hourly = pd.DataFrame({
    "Date": pd.to_datetime(data["hourly"]["time"]),
    "Temperature": data["hourly"]["temperature_2m"],
    "Humidity": data["hourly"]["relative_humidity_2m"],
    "Pressure": data["hourly"]["surface_pressure"]
})

df_hourly.set_index("Date", inplace=True)
df = df_hourly.resample('D').mean()
df.dropna(inplace=True)

df['Day_Sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
df['Day_Cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

for lag in [1, 2, 3]:
    df[f'Temp_Lag_{lag}'] = df['Temperature'].shift(lag)
    df[f'Hum_Lag_{lag}'] = df['Humidity'].shift(lag)

df['Target_Temperature'] = df['Temperature'].shift(-1)
df.dropna(inplace=True)

feature_cols = ['Humidity', 'Pressure', 'Day_Sin', 'Day_Cos', 'Temp_Lag_1', 'Temp_Lag_2', 'Temp_Lag_3', 'Hum_Lag_1', 'Hum_Lag_2', 'Hum_Lag_3']
X = df[feature_cols]
y = df['Target_Temperature']

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

joblib.dump(model, "weather_model.pkl")