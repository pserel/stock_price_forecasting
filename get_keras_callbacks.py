# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:30:35 2019

@author: SERELPA1
"""

# Fitting RNN to training set using Keras Callbacks. Read Keras callbacks docs for more info.
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
def get_keras_callbacks(es_patience = 30,
                        rlr_factor = 0.75,
                        rlr_patience = 5):
    es = EarlyStopping(monitor = 'val_loss',
                       min_delta = 1e-10,
                       patience = es_patience,
                       verbose = 1)
    rlr = ReduceLROnPlateau(monitor = 'val_loss',
                            factor = rlr_factor,
                            patience = rlr_patience, 
                            verbose = 1)
    mcp = ModelCheckpoint(filepath = 'weights.h5',
                          monitor = 'val_loss',
                          verbose = 1,
                          save_best_only = True, 
                          save_weights_only = True)
    tb = TensorBoard('logs')
    
    return([es, rlr, mcp, tb])
