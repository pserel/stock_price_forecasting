# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 23:22:03 2019

@author: SERELPA1
"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler

def feature_scaling(ds_train):
    sc = StandardScaler()
    ds_train_scaled = sc.fit_transform(ds_train)
 
    sc_predict = StandardScaler()
    sc_predict.fit_transform(ds_train[:,0:1])
    
    return(sc, sc_predict, ds_train_scaled)