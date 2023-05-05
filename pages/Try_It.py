import pandas as pd
import numpy as np
import streamlit as st
import joblib
from PIL import Image
from sklearn.preprocessing import StandardScaler

from streamlit_folium import folium_static
import folium
from datetime import datetime, time
from PIL import Image

@st.cache_data()
def load(scaler_path, model_path):
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model

def inference(row, cols, scaler, model):
    data = pd.DataFrame([row], columns = cols) 
    data = scaler.transform(data)
 
    duration = model.predict(data)[0]
    
    return duration

column = ['vendor_id', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag', 'year', 'month', 'date', 'hour']

# Set initial location to San Francisco
map = folium.Map(location=[40.8, -74.0], zoom_start=12)

# Render the map using folium_static instead of st.map()
st.title("Try It")
st.write("In this page, You can simulate as a passenger who will order a yellow taxi in New York, and find out the estimated trip duration. You can select on the side bar, your own time, location, and the other variable of the order of the taxi, then click Predict Trip Duration")
img = Image.open(r"./image/passengers.jpg")
st.image(img, use_column_width = True)
st.write("______________________________________________________")
st.write('Find more about latitude and longitude location on the map')

map.add_child(folium.LatLngPopup())

popup = folium.LatLngPopup()
folium_static(map)

vendor = st.sidebar.slider("Select Vendor", 1, 3, 1, 1)

pickup_lat = st.sidebar.number_input("Pickup Latitude", 34.4, 51.9, 40.8, 0.1)
pickup_long = st.sidebar.number_input("Pickup Longitude", -121.9, -61.3, -74.0, 0.1)
dropoff_lat = st.sidebar.number_input("Dropoff Latitude", 32.2, 43.9, 40.8, 0.1)
dropoff_long = st.sidebar.number_input("Dropoff Longitude", -121.9, -61.3, -74.0, 0.1)

date_pickup = st.sidebar.date_input("Date pick-up", datetime.now().date())
time_pickup = st.sidebar.time_input("Date pick-up", datetime.now().time())

# passenger = st.sidebar.slider("Number of Passenger", 0, 7, 1, 1)

store = st.sidebar.selectbox("Data Store", ('Online','Offline'))

if (st.sidebar.button('Predict Trip Duration')):
    if(store == "Online"):
        store_n = 1
    else:
        store_n = 0
    new_data = [vendor, pickup_long, pickup_lat, dropoff_long, dropoff_lat, store_n, date_pickup.year, date_pickup.month, date_pickup.day, time_pickup.hour]
    scaler, model = load('model/scaler.joblib', 'model/DecisionTreeRegressor().joblib')
    duration = inference(new_data, column, scaler, model)
    st.write("________________________________________")
    st.header('Predicting Result')
    st.subheader('Detail Information')
    st.write(f"""<div style='text-align:justify'>
    <b>Date Time Pickup</b>
    Date: {date_pickup} <br>
    Time: {time_pickup.strftime('%H:%M:%S')} <br>
    <b>Pickup Location</b><br>
    Latitude: {pickup_lat} <br>
    Longitude: {pickup_long} <br>
    <b>Dropoff Location</b>
    Latitude: {dropoff_lat} <br>
    Longitude: {dropoff_long} <br>
    <b> Other Information </b> <br>
    Vendor: {vendor} <br>
    Data Store: {store} <br>
    </div>""", unsafe_allow_html=True)
    st.subheader("Trip Duration Predicting")
    st.write(f'Trip duration: {duration.round(0)} second or {(duration/60).round(0)} minutes')