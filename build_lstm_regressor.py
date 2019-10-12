# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:51:15 2019

@author: SERELPA1
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def build_lstm_regressor(n_lstm_layers = 6, drop_out = 0.5, n_units = 100,
                         optimizer = 'adam', n_past = 25, n_future = 25):
    # Initializing the RNN
    regressor = Sequential()
 
    # Adding first LSTM layer and Drop out Regularization
    regressor.add(LSTM(units=n_units, return_sequences=True, input_shape=(n_past, 5)))
    regressor.add(Dropout(drop_out))
    
    # Adding more layers
    for l in range(n_lstm_layers - 2): 
        regressor.add(LSTM(units=n_units, return_sequences=True))
        regressor.add(Dropout(drop_out))


    regressor.add(LSTM(units = n_units))
    regressor.add(Dropout(drop_out))

    # Output layer
    regressor.add(Dense(units=n_future))
 
    # Compiling the RNN
    regressor.compile(optimizer=optimizer, loss="mean_squared_error")  # Can change loss to mean-squared-error if you require.
    
    return regressor
