# ------------------------- IMPORTS -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import load_model
import datetime
import pydeck as pdk

# ------------------------- LOAD DATA & MODEL -------------------------
df = pd.read_csv("comined_5cities.csv", parse_dates=["Date"])
df['city'] = df['city'].astype(str)

# Feature Engineering
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['day_of_week'] = df['Date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

# City-specific holidays
def get_city_holidays(date, city):
    try:
        city = str(city).lower()
        if city in ["abudhabi", "dubai", "sharjah"]:
            return 1 if date in holidays.UnitedArabEmirates() else 0
        elif city == "qatar":
            return 1 if date in holidays.Qatar() else 0
        elif city == "oman":
            return 1 if date in holidays.Oman() else 0
        else:
            return 0
    except:
        return 0

df['is_holiday'] = df.apply(lambda x: get_city_holidays(x['Date'], x['city']), axis=1)

# Encode city
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

# Define features
weather_features = [
    'temperature_2m_max', 'temperature_2m_mean', 'temperature_2m_min',
    'apparent_temperature_max', 'apparent_temperature_mean', 'apparent_temperature_min',
    'precipitation_sum', 'rain_sum', 'showers_sum', 'snowfall_sum',
    'precipitation_hours', 'weather_code', 'wind_speed_10m_max', 'wind_gusts_10m_max',
    'et0_fao_evapotranspiration'
]

calendar_features = ['day','month','day_of_week','is_weekend','is_holiday','city_encoded']
features = weather_features + calendar_features
target = "Consumption_GWh"

# Scalers
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
df_scaled = df.copy()
df_scaled[features] = scaler_X.fit_transform(df_scaled[features])
df_scaled[[target]] = scaler_y.fit_transform(df_scaled[[target]])

from tensorflow.keras.models import load_model
import tensorflow as tf

# Correct way to load the model with 'mse' metric/loss
model = load_model("lstm_multi_city_model.h5", custom_objects={'mse': tf.keras.losses.MeanSquaredError})


# ------------------------- STREAMLIT APP -------------------------
st.set_page_config(page_title="Energy Demand Forecast", layout="wide")
st.title("⚡ Multi-City Energy Demand Forecast")

# --- User Inputs ---
city_selected = st.selectbox("Select City", df['city'].unique())

forecast_days = st.slider("Forecast Days", 1, 14, 7)

st.subheader("Weather Inputs (for forecast)")
weather_inputs = {}
for f in weather_features:
    min_val = float(df[f].min())
    max_val = float(df[f].max())
    if min_val == max_val:  # avoid Streamlit slider error
        max_val += 1
    weather_inputs[f] = st.slider(f, min_val, max_val, float((min_val+max_val)/2))

# ------------------------- FORECAST FUNCTION -------------------------
time_steps = 7

def forecast_city(city_name, forecast_days, custom_weather):
    city_df = df[df['city']==city_name].sort_values("Date").reset_index(drop=True)
    
    # Initialize last sequence window
    last_window = city_df[features].iloc[-time_steps:].copy()
    
    # Keep calendar + city_encoded same
    last_window['city_encoded'] = city_df['city_encoded'].iloc[-1]
    
    # Collect predictions
    predictions = []
    last_date = city_df['Date'].iloc[-1]

    for i in range(forecast_days):
        # Update weather features for the last day in window
        for col, val in custom_weather.items():
            last_window.iloc[-1][col] = val
        
        # Update calendar features for next day
        next_date = last_date + pd.Timedelta(days=1)
        last_window.iloc[-1]['day'] = next_date.day
        last_window.iloc[-1]['month'] = next_date.month
        last_window.iloc[-1]['day_of_week'] = next_date.dayofweek
        last_window.iloc[-1]['is_weekend'] = int(next_date.dayofweek in [5,6])
        last_window.iloc[-1]['is_holiday'] = get_city_holidays(next_date, city_name)
        
        # Scale
        last_window_scaled = scaler_X.transform(last_window).reshape((1, time_steps, len(features)))
        
        # Predict
        pred_scaled = model.predict(last_window_scaled)[0]
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).reshape(-1)[0]
        predictions.append(pred)
        
        # Slide window: drop first row, append new predicted row (with previous day features)
        new_row = last_window.iloc[-1].copy()
        last_window = pd.concat([last_window.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
        last_date = next_date
    
    # Build forecast DataFrame
    future_dates = [city_df['Date'].iloc[-1] + pd.Timedelta(days=i) for i in range(1, forecast_days+1)]
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Consumption_GWh": predictions
    })
    
    return forecast_df


# ------------------------- RUN FORECAST -------------------------
forecast_df = forecast_city(city_selected, forecast_days, weather_inputs)

# ------------------------- DISPLAY CHART -------------------------
st.subheader("Forecasted Energy Consumption")
st.line_chart(forecast_df.set_index("Date")["Predicted_Consumption_GWh"])

# ------------------------- DISPLAY MAP -------------------------
st.subheader("City Location")
city_coords = {
    "abudhabi": (24.4539, 54.3773),
    "dubai": (25.276987, 55.296249),
    "sharjah": (25.3463, 55.4209),
    "qatar": (25.3548, 51.1839),
    "oman": (23.588, 58.3829)
}
lat, lon = city_coords[city_selected.lower()]

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=8,
        pitch=0,
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=pd.DataFrame([{"lat": lat, "lon": lon}]),
            get_position='[lon, lat]',
            get_color='[255, 0, 0, 200]',
            get_radius=5000,
        )
    ]
))
