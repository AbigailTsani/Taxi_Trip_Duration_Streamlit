import streamlit as st
from PIL import Image

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from joblib import dump, load
import pickle
import os

def modeling_scaled(option_var):
  st.subheader(f"Residual Plot of {option_var} Prediction")
  img = Image.open(f"./image/{option_var}.jpg")
  st.image(img, use_column_width = True)

  st.subheader('Interpretation of Result in Test Dataset')
  if str(option_var) == 'DecisionTreeRegressor()':
    st.write(f"""<div style='text-align:justify'>
    <b>{option_var}</b><br>
    <ol>
    <li> The distribution of the residual is <b> a normal distribution </b>, which mean the model is a good regression model because the assumption of the model is sufficient </li>
    <li> RMSE is 1.3x higher than MAE, which mean there are several data that are predicted is <b> close from the ground truth </b></li>
    <li> MAPE can be interpret that the error of predicting is 0.5, which mean <b> not bed because the error is not high</b></li>
    <li> R2 score is 0.6 have higher score than other model, which mean the model <b>predict better</b></li>
    </ol>
    </div>""", unsafe_allow_html=True)
  else:
    st.write(f"""<div style='text-align:justify'>
    <b>{option_var}</b><br>
    <ol>
    <li> The distribution of the residual is <b> not a normal distribution</b>, which mean the model is not a good regression model because the assumption of the model is not sufficient </li>
    <li> RMSE is 1.23x higher than MAE, which mean there are several data that are predicted is to <b>far close the ground truth </b></li>
    <li> MAPE can be interpret that the error of predicting is 0.92, which mean <b>not good because have high error </b></li>
    <li> R2 score is 0.02 not good, cause the score indicate that the model has <b>low of fit </b>of the data, which mean the model <b>doesn't predict well</b></li>
    </ol>
    </div>""", unsafe_allow_html=True)

st.title("Model")
st.write("In this page, you can see the results of the evaluation of the model made. Please select the model on the sidebar")
image1 = Image.open(r"./image/regression.jpg")
st.image(image1, use_column_width = True)
st.subheader("Type of Model")
model_type = """
There are 4 type of model to be used:
<ol>
  <li> Linear Regression </li>
  <li> Ridge Regression </li>
  <li> Lasso Regression </li>
  <li> Decision Tree Regressor </li>
</ol>
"""
st.write(f"<div style='text-align:justify'>{model_type}</div>", unsafe_allow_html=True)

st.subheader("Data for modeling")
data_model = """
Before doing modeling, data preprocessing is carried out, 
where testing has been carried out regarding the best preprocessing data. 
The data used for the model has gone through preprocessing:
<ol>
  <li> Handling missing value, duplicated data, and outlier </li>
  <li> Drop unnessesary variable </li>
  <li> Scaling data using Standard Scaler </li>
</ol>
"""
st.write(f"<div style='text-align:justify'>{data_model}</div>", unsafe_allow_html=True)
st.write("_________________________________________________")
option_var = st.sidebar.selectbox("Choose the model", [LinearRegression(), Ridge(), Lasso(), DecisionTreeRegressor()])

modeling_scaled(option_var)