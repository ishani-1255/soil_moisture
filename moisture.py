#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 12:18:55 2024

@author: ishikaishani
"""

import numpy as np
import pickle 
from catboost import CatBoostRegressor
import streamlit as st
loaded_model = pickle.load(open('soilmoisturemodel.sav', 'rb'))

def soil_moisture_prediction(input_data):

    # change the input data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def main():
    st.title('Soil_Moisture_Prediction')
    
    CropDays = st.text_input('No of Crop Days :')
    Temperature = st.text_input('Temperature :')
    Humidity = st.text_input('Humidity :' )
    
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Soil_Moisture_Result'):
        diagnosis = soil_moisture_prediction([CropDays, Temperature, Humidity])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
