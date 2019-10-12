# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 23:26:13 2019

@author: SERELPA1
"""
# Creating an accuracy measuring function for time series regression
import numpy as np

def get_mape(y_true, y_pred):
    mape = np.mean(np.abs(y_pred/y_true - 1))
    mape = round(mape, 3)
    return(mape)