# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 16:50:45 2025

@author: Muizzah
"""

import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import sklearn
import pickle 
import lightgbm
import calendar
from datetime import time
import joblib
import os


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "Electricity_model.joblib")
loaded_model = joblib.load(MODEL_PATH)


def electricity(features):
    feature_names = ['Global_reactive_power', 'Voltage', 'Global_intensity',
       'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'year',
       'dayofweek', 'month', 'day', 'Time_sec'] 
    
    features_dataframe = pd.DataFrame([features], columns=feature_names)
    Time_sec = st.session_state["Timess"].hour*3600 + st.session_state["Timess"].minute*60 + st.session_state["Timess"].second
    features_dataframe['Time_sin'] = np.sin(2 * np.pi * Time_sec / 86400)
    features_dataframe['Time_cos'] = np.cos(2 * np.pi * Time_sec / 86400)
    features_dataframe['month_sin'] = np.sin(2 * np.pi * (features_dataframe['month'] -1) / 12)
    features_dataframe['month_cos'] = np.cos(2 * np.pi * (features_dataframe['month'] -1) / 12)
    days_in_month = calendar.monthrange(features_dataframe['year'].any(), features_dataframe['month'].any())[1]
    day_norm = (features_dataframe['day'] - 1.0) / days_in_month
    features_dataframe['day_sin'] = np.sin(2*np.pi*day_norm)
    features_dataframe['day_cos'] = np.cos(2*np.pi*day_norm)
    features_dataframe['lag_1'] = features_dataframe['lag_5'] = features_dataframe['lag_60'] = features_dataframe['lag_1440'] = st.session_state["number"]
    features_dataframe['rolling_15min'] = features_dataframe['rolling_60min'] = features_dataframe['rolling_1hr'] = st.session_state["number"]
    features_dataframe = features_dataframe.drop(["Time_sec", 'month', 'day'], axis=1)
    
    new_feature_order = ['Global_reactive_power', 'Voltage', 'Global_intensity',
       'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'year',
       'dayofweek', 'Time_sin', 'Time_cos', 'month_sin', 'month_cos',
       'day_sin', 'day_cos', 'lag_1', 'lag_5', 'lag_60', 'lag_1440',
       'Rollling_15min', 'Rolling_60min', 'Rolling_1hr']
    features_dataframe = features_dataframe[new_feature_order]
    
    
    prediction = loaded_model.predict(features_dataframe)
    return prediction
 

def main():
    st.title("Household Electric Power Consumption Prediction Web App")
    Global_reactive_power = 0.13 
    Voltage = st.number_input("Enter voltage value", min_value=0, max_value=300, value=220)
    Global_intensity = st.slider("Enter Global Intensity (A)", 0.4,47.0,5.3)  
        
    Sub_metering_1 = st.number_input("Energy used by kitchen appliances (kWh)", min_value=0, max_value=1000, value=0)
    Sub_metering_2 = st.number_input("Energy used by laundry room appliances (kWh)", min_value=0, max_value=1000, value=0)
    Sub_metering_3 = st.number_input("Energy used by water heater & AC (kWh)", min_value=0, max_value=1000, value=0)
    Time_sec = st.time_input("Select a time:", value=time(12, 0), key='Timess')
    selected_date = st.date_input("Select a date")
    year = selected_date.year 
    dayofweek = selected_date.weekday()
    month = selected_date.month 
    day = selected_date.day
    recent_value = st.number_input("Most recent Global Active Power (kW):", value=1.5, key="number")
    
    
    Electricity = ''
    if st.button('Electricity Consumption result'):                                  
        Electricity = electricity([Global_reactive_power, Voltage, Global_intensity, Sub_metering_1, Sub_metering_2,
        Sub_metering_3, year, dayofweek, month , day, Time_sec])

    st.success(Electricity) 
                                  

if __name__ == '__main__':
    main()







