import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import holidays
import os

# Page config
st.set_page_config(
    page_title="⚡ Multi-City Energy Forecast",
    page_icon="💡",
    layout="wide"
)

# === CONFIG ===
CITIES = ["dubai", "abudhabi", "sharjah", "oman", "qatar"]
CITY_TO_COUNTRY = {
    'dubai': 'AE', 'abudhabi': 'AE', 'sharjah': 'AE',
    'oman': 'OM', 'qatar': 'QA'
}
COORDS = {
    'dubai': [55.2708, 25.2048],
    'abudhabi': [54.3667, 24.4667],
    'sharjah': [55.4211, 25.3575],
    'oman': [58.4059, 23.5859],
    'qatar': [51.5333, 25.2833]
}

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('energy_xgb_model.pkl')
        scaler = joblib.load('scaler_X.pkl')
        features = joblib.load('feature_columns.pkl')
        metrics = joblib.load('model_metrics.pkl')
        return model, scaler, features, metrics
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None, None

def create_features(date, city, temp_mean, is_holiday_override=None):
    """Generate feature vector for a single date + city"""
    date_obj = pd.to_datetime(date)
    weekday = date_obj.weekday()
    month = date_obj.month
    day = date_obj.day
    year = date_obj.year
    
    # Determine actual holiday
    country = CITY_TO_COUNTRY.get(city.lower(), 'AE')
    if is_holiday_override is not None:
        is_holiday = int(is_holiday_override)
    else:
        h = holidays.country_holidays(country, years=[year])
        is_holiday = 1 if date_obj.date() in h else 0

    # >>> FIX: Correct way to get last day of month <<<
    next_month = date_obj.replace(day=28) + timedelta(days=4)
    last_day_of_month = (next_month - timedelta(days=next_month.day)).day

    # Weather (deterministic with seed for demo)
    np.random.seed(hash(str(date) + city) % (2**32))
    features = {
        "week_of_year": date_obj.isocalendar()[1],
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "dayofweek_sin": np.sin(2 * np.pi * weekday / 7),
        "dayofweek_cos": np.cos(2 * np.pi * weekday / 7),
        "is_holiday": is_holiday,
        "is_weekend": 1 if weekday >= 5 else 0,
        "day_of_month": day,
        "quarter": (month - 1) // 3 + 1,
        "year": year,
        "is_month_start": 1 if day == 1 else 0,
        "is_month_end": 1 if day == last_day_of_month else 0,  # ✅ FIXED!
        'temperature_2m_max': temp_mean + np.random.uniform(2, 4),
        'temperature_2m_mean': temp_mean,
        'temperature_2m_min': temp_mean - np.random.uniform(2, 4),
        'apparent_temperature_max': temp_mean + np.random.uniform(1, 3),
        'apparent_temperature_mean': temp_mean,
        'apparent_temperature_min': temp_mean - np.random.uniform(1, 3),
        'precipitation_sum': np.random.uniform(0, 2),
        'rain_sum': np.random.uniform(0, 2),
        'showers_sum': 0.0,
        'snowfall_sum': 0.0,
        'precipitation_hours': np.random.uniform(0, 6),
        'weather_code': 1,
        'wind_speed_10m_max': np.random.uniform(8, 20),
        'wind_gusts_10m_max': np.random.uniform(15, 30),
        'et0_fao_evapotranspiration': max(2.0, temp_mean / 8 + np.random.normal(0, 0.5))
    }
    
    # Add city dummies
    for c in CITIES:
        features[f'city_{c}'] = 1 if c == city else 0
        
    return features

def predict_single(model, scaler, features, feature_columns):
    df = pd.DataFrame([features])
    df = df.reindex(columns=feature_columns, fill_value=0)
    scaled = scaler.transform(df)
    return float(model.predict(scaled)[0])

def generate_forecast(model, scaler, feature_columns, city, start_date, days, temp_mean):
    dates = [start_date + timedelta(days=i) for i in range(1, days+1)]
    preds = []
    for d in dates:
        feat = create_features(d, city, temp_mean)
        pred = predict_single(model, scaler, feat, feature_columns)
        preds.append({'Date': d, 'Consumption_GWh': pred})
    return pd.DataFrame(preds)

def create_map(forecast_df, map_style="open-street-map"):
    """Create interactive map showing demand across cities"""
    if forecast_df is None or len(forecast_df) == 0:
        return None
    
    # Use latest forecast value for all cities
    latest_demand = forecast_df['Consumption_GWh'].iloc[-1] if len(forecast_df) > 0 else 100.0
    
    map_data = []
    for city, (lon, lat) in COORDS.items():
        map_data.append({
            'City': city.capitalize(),
            'Latitude': lat,
            'Longitude': lon,
            'Demand_GWh': latest_demand
        })
    
    df_map = pd.DataFrame(map_data)
    min_val, max_val = df_map['Demand_GWh'].min(), df_map['Demand_GWh'].max()
    df_map['Size'] = 10 + 20 * (df_map['Demand_GWh'] - min_val) / (max_val - min_val + 1e-6)  # avoid div by zero

    fig = px.scatter_mapbox(
        df_map,
        lat="Latitude",
        lon="Longitude",
        size="Size",
        color="Demand_GWh",
        hover_name="City",
        color_continuous_scale="Viridis",
        size_max=30,
        zoom=5,
        height=400,
        title=f"Energy Demand Map — {forecast_df['Date'].iloc[-1].strftime('%Y-%m-%d')}"
    )
    fig.update_layout(mapbox_style=map_style, margin={"r":0,"t":40,"l":0,"b":0}, showlegend=False)
    return fig

# === MAIN APP ===
def main():
    st.title("⚡ Multi-City Energy Demand Forecast")
    st.markdown("Real-time forecasting for middle eastren citties such as abudhabi, UAE, Oman, sharjah & Qatar with geo maps!")

    model, scaler, feature_columns, metrics = load_artifacts()
    if model is None:
        st.stop()

    # Sidebar
    st.sidebar.header("🎛️ Controls")
    selected_city = st.sidebar.selectbox("City", CITIES)
    forecast_date = st.sidebar.date_input("Date", datetime.now().date() + timedelta(days=1))
    temperature = st.sidebar.slider("Mean Temp (°C)", 10.0, 45.0, 28.0)
    is_holiday = st.sidebar.checkbox("Force Holiday", False)

    # Check real holiday
    country = CITY_TO_COUNTRY[selected_city]
    h = holidays.country_holidays(country, years=[forecast_date.year])
    if forecast_date in h:
        st.sidebar.success(f"🎉 Official holiday in {selected_city}!")

    # Real-time prediction
    if st.sidebar.button("🔮 Predict Demand"):
        feat = create_features(forecast_date, selected_city, temperature, is_holiday)
        pred = predict_single(model, scaler, feat, feature_columns)
        
        st.subheader(f"Predicted Demand: {pred:.2f} GWh")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title=f"{selected_city.title()} on {forecast_date}",
            gauge={'axis': {'range': [80, 160]}, 'bar': {'color': "royalblue"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    # Generate forecasts on-demand
    st.markdown("---")
    st.subheader("📈 Generate Forecast Series")
    forecast_days = st.selectbox("Forecast Horizon", [7, 14, 30])
    
    if st.button(f"Generate {forecast_days}-Day Forecast"):
        start = datetime.now().date()
        df_fc = generate_forecast(model, scaler, feature_columns, selected_city, start, forecast_days, temperature)
        
        # Plot
        fig = px.line(df_fc, x='Date', y='Consumption_GWh', markers=True,
                      title=f"{forecast_days}-Day Forecast for {selected_city.title()}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary
        st.info(f"Average: {df_fc['Consumption_GWh'].mean():.2f} GWh | "
                f"Peak: {df_fc['Consumption_GWh'].max():.2f} GWh")

        # 🗺️ Show Map
        st.markdown("---")
        st.subheader("🗺️ Geographic Demand Map")
        col1, col2 = st.columns([3, 1])
        with col1:
            map_style = st.selectbox("Map Style", [
                "open-street-map", 
                "carto-positron", 
                "carto-darkmatter", 
                "stamen-terrain"
            ])
        with col2:
            st.write("")
            st.write("")
            st.write("")
            if st.button("🔄 Refresh Map"):
                pass  # just re-render

        map_fig = create_map(df_fc, map_style)
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True)
            st.caption("All cities shown with latest forecasted demand level.")

    # Model info
    st.markdown("---")
    st.subheader("📊 Model Performance")
    if metrics:
        st.metric("R² Score", f"{metrics['r2']:.3f}")
        st.metric("MAE", f"{metrics['mae']:.2f} GWh")
        st.metric("RMSE", f"{metrics['rmse']:.2f} GWh")

if __name__ == "__main__":
    main()
