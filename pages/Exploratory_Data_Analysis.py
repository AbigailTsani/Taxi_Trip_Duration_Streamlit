import streamlit as st
from PIL import Image
import folium
from streamlit_folium import folium_static

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df= pd.read_csv(r"./data/data_100.csv")

st.title("Exporatory Data Analysis")
st.write("What the data tells us?")

option_var = st.sidebar.selectbox("Choose the variable you want to know more", ["Pick-up Location", "Drop-off Location", "Pick-up_Month", "Pick-up_Date", "Pick-up_Hour", "Number_of_Passenger", "Vendor_ID", "Store_and_forward_flag", "Trip Duration"])

st.header(option_var)
if option_var == "Pick-up Location":
    st.write("The following are 100 random data distributions for pickup locations")
    pick = folium.Map(location=[df["pickup_latitude"].mean(), df["pickup_longitude"].mean()], zoom_start=13)
    for i in range(0, 100):
        folium.Marker(
            location=[df["pickup_latitude"].iloc[i], df["pickup_longitude"].iloc[i]],
            # icon=folium.Icon(icon='cloud')
        ).add_to(pick)
    folium_static(pick)
    st.write("Most of the passenger pickup in the new york city area and surrounding areas")

elif option_var == "Drop-off Location":
    st.write("The following are 100 random data distributions for dropoff locations")
    drop = folium.Map(location=[df["dropoff_latitude"].mean(), df["dropoff_longitude"].mean()], zoom_start=13)
    for i in range(0, 100):
        folium.Marker(
            location=[df["dropoff_latitude"].iloc[i], df["dropoff_longitude"].iloc[i]],
            # icon=folium.Icon(icon='cloud')
        ).add_to(drop)
    folium_static(drop)
    st.write("Most of the passenger drop off in the new york city area and surrounding areas")

elif option_var == "Trip Duration":
    img = Image.open(r"./image/trip_duration.jpg")
    st.image(img, use_column_width = True)

    st.write("The distribution of trip duration data is positively skewed where the trip duration value is most often in the range of 250-500 seconds or 4-9 minutes in 1 trip")


else:
    img = Image.open(f"./image/{option_var}.jpg")
    st.image(img, use_column_width = True)
    if option_var == "Number_of_Passenger":
        st.write(f"Most Number_of_Passenger is 1, with total frequency of the most is 858,014 or 79.52% of the data")
    elif option_var == "Vendor_ID":
       st.write("Every vendor have almost the same order")
    elif option_var == "Store_and_forward_flag":
       st.write("All trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server")
    elif option_var == "Pick-up_Month":
      st.write("Every month the number of orders is almost the same. So there is no trend of the month")
    elif option_var == "Pick-up_Date":
      st.write("Every date the number of orders is almost the same, So there is no trend of the date")
    else:
        st.write("Most order is at time 18.00 - 21.00")


