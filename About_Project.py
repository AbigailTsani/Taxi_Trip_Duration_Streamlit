
import streamlit as st
from PIL import Image

st.write("""
    <style>
    /* CSS styles */
    body {
        background-color: #F5F5F5;
        font-family: Arial, sans-serif;
        font-size: 1.2em;
        padding: 20px;
    }
    h1 {
        color: #008080;
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 40px;
    }
    p {
        line-height: 1.5;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Taxi Trip Duration Prediction")
img = Image.open(r"./image/taxi.jpg")
st.image(img, use_column_width = True)

#---Overview---
st.header("Overview")
overview_text = """
This project will discuss the prediction of trip duration using a <b>regression model</b>, 
more specifically, <b>new york taxi trip duration</b>. This project retrieves data from 
<a href = https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data>https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data</a>. 
The objective of this project is <b>to improve customer service and 
encorage the customer </b>to use yellow medallion taxis by providing estimated travel time. 
Based on modeling, the <b>decision tree regressor model</b> was obtained as the best model with 
a mean absolute error and root mean square error of 165 and 218 or it can be said 
to have an <b> estimated average error of 2.8 minutes </b>.
<br>
<br>
"""
st.write(f"<div style='text-align:justify'>{overview_text}</div>", unsafe_allow_html=True)

#---Background---
st.header("Background")

#---Company background---
st.subheader("Company Background")
background_text = """
New York City Taxi and Limousine Commission, created in 1971, is the agency responsible for licensing 
and regulating New York City's Medallion (Yellow) taxi cabs, 
for-hire vehicles (community-based liveries, black cars and luxury limousines) , 
commuter vans, and paratransit vehicles. Over 200,000 TLC licensees total approximately 
1,000,000 trips per day.
<br>
<br>
"""
st.write(f"<div style='text-align:justify'>{background_text}</div>", unsafe_allow_html=True)

#--- Current State---
st.subheader("Current State")
st.write("**Average Daily Trip Volume Proportion by Sector**")
image1 = Image.open(r"./image/taxi_problem1.jpg")
legend1 = Image.open(r"./image/legend1.jpg")
st.image(image1, use_column_width = True)
st.image(legend1)
problem_text1 = """
There is a decrease in the proportion of users of yellow medallion taxis and 
green street hail livery, otherwise there has been an increase in 
high volume for-hire vehicles (HVFHV) and traditional for-hire vehicles (TFHV).
More clearly can be seen in the image below regarding the trend of use of every sector in New York City taxis.
<br>
<br>
"""
st.write(f"<div style='text-align:justify'>{problem_text1}</div>", unsafe_allow_html=True)

st.write("**Trip Counts Per Day**")
image2 = Image.open(r"./image/taxi_problem2.jpg")
st.image(image2, use_column_width = True)
problem_text2 = """
There has been a decrease in the use of yellow medallion taxis, 
and conversely there has been a very significant increase in 
the use of high volume for-hire vehicle (HVFHV).
<br>
<br> Source: <a> https://www.nyc.gov/assets/tlc/downloads/pdf/2020-tlc-factbook.pdf </a>
<br>
<br>
"""
st.write(f"<div style='text-align:justify'>{problem_text2}</div>", unsafe_allow_html=True)

#--- Business Question---
st.subheader("Reason Behide This Behaviour")
reason1_text = """
1. <b>Convenience</b>: Request a ride through a mobile app, track the driver's location in real time, and easier to budget for the rides. <br>
2. Transparency: See the driver's profile, can leave their own rating and feedback, help increasing secure and confident. <br>
3. Availability: More readily (especially in suburban and rural areas). <br>
4. Cost: Depending on the location and the specific ride. <br>
<br> Focus on the convenience part, namely the main advantage of HVFHV is that it is easier for the customer to budget for the rides, 
where this determination is made in advance by knowing the trip duration. <br>
<br>
"""
st.write(f"<div style='text-align:justify'>{reason1_text}</div>", unsafe_allow_html=True)
st.subheader("The Advantages of Predicting Trip Duration")
reason2_text = """
1. Better Route Planning: lead to more efficient and faster trips by taking into account the expected travel time and any potential delays. <br>
2. Improved Scheduling and Dispatching: assign taxis more effectively, based on expected travel time, helping reduce wait times for passengers and increase the efficiency of the taxi fleet. <br>
3. Increased Safety: reducing the likelihood of drivers speeding or taking unnecessary risks to make up for lost time. <br>
4. Better Customer Satisfaction: reducing uncertainty and anxiety for passengers. <br>
<br>
"""
st.write(f"<div style='text-align:justify'>{reason2_text}</div>", unsafe_allow_html=True)

#--- Objective---
st.header("Objective")
obj_text = """
Create machine learning models that can estimate trip duration with the lowest error based on historical data
<br><br>
"""
st.write(f"<div style='text-align:justify'>{obj_text}</div>", unsafe_allow_html=True)

#--- Method---
st.header("Methods")
method_text = """
Cross Industry Standard Process for Data Mining (CRISP-DM) Metodology. <br>
Model: Regression Model <br>
1. Linear Regression <br>
2. Ridge Regression <br>
3. Lasso Regression <br>
4. Decision Tree Regression <br>
<br>
"""
st.write(f"<div style='text-align:justify'>{method_text}</div>", unsafe_allow_html=True)

#---Result---
st.header("Results")
result_text = """
Best model: Decision Tree Regression <br>
Hyperparameter: max_depth = 15, max_feature = 15 <br>
RMSE = 217.7 <br>
MAE = 164.6 <br>
An error in the model predicts a trip duration of only 2.8 minutes. <br><br>
The following is the feature importances value of each feature that 
most influences the trip duration: <br>
1. pickup_latitude = 0.41 <br>
2. dropoff_latitude = 0.24 <br>
3. pickup_longitude = 0.15 <br>
4. dropoff_longitude = 0.12 <br> <br>
It can be interpreted that the pick-up and drop-off points are the features 
that most influence the trip duration. As for vendors and store and forward flags 
as well as pick-up times, they do not have a large influence on the trip duration
"""
st.write(f"<div style='text-align:justify'>{result_text}</div>", unsafe_allow_html=True)

