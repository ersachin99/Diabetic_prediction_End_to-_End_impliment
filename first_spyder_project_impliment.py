# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:26:14 2024

@author: sachin gavalkar
"""

import numpy as np
import pandas as pd
import pickle

# load the saving model
loaded_model=pickle.load(open("D:/PROJECT ML/Project Implimnet/trained_model.sav", "rb"))


## making the predictiom system

input_data=(5,166,72,19,175,25.8,0.587,51)

# changing the input data to numpy aarra6y
input_data_as_numpy_array=np.asarray(input_data)


# reshape the array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)

print(prediction)


if (prediction[0]==0):
    print('The Person Is Not Diabetic')

else:
    print('The Person Is Diabetic')
