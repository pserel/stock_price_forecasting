# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:19:27 2019

@author: SERELPA1
"""

# Creating a data structure with n_past timesteps and n_future outputs
import numpy as np

def lstm_data_structure(ds, n_past = 25, n_future = 25, n_features = 1):
    
    X_train = []
    y_train = []
 
    for i in range(n_past, len(ds) - n_future + 1):
        X_train.append(ds[i - n_past:i, 0:n_features])
        y_train.append(ds[i:i + n_future, 0])
 
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    return(X_train, y_train)