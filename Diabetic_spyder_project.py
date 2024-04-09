# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:39:11 2024

@author: sachin gavalkar
"""

import streamlit as st
import numpy as np
import pickle



 #load the saving model
loaded_model=pickle.load(open("D:/PROJECT ML/Project Implimnet/trained_model.sav", "rb"))
                              
                              
                              
# now creating a function for predictions

def diabetic_prediction(input_data):
    
    

    # changing the input data to numpy aarra6y
    input_data_as_numpy_array=np.asarray(input_data)


    # reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)

    print(prediction)


    if (prediction[0]==0):
        return 'The Person Is Not Diabetic'

    else:
        return 'The Person Is Diabetic'
    

def main():
    
    # giving the title
    st.title('Diabetic Prediction')
    
    # create a input data for 9 columns 
    # getting the input data from the user 
    
    Pregnancies=st.text_input('Number of Pregancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('Blood Pressure Value')
    SkinThickness=st.text_input('SkinThickness of User')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI Value')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function  Values')
    Age=st.text_input('Age of the Person')





    # code for the prediction
    diagnosis=''
    
    # create a button for diagnosis for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis=diabetic_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    
    st.success(diagnosis)
    

    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
                                
                              