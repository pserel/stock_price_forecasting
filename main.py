#import os
#abspath = os.path.abspath('') ## String which contains absolute path to the script file. Also getcwd()?
#os.chdir(abspath) ## Setting up working directory

# "".join([getcwd(),'\\scripts'])

# Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


n_future = 25  # Number of days you want to predict into the future
n_past = 25  # Number of past days you want to use to predict the future
n_features = 5 # Number of features to use
 
# Importing Data Set
dataset = pd.read_csv('acn.csv')

dataset_train = dataset.iloc[:len(dataset) - n_future - n_past, :]
dataset_test = dataset.iloc[len(dataset) - n_future - n_past:, :]
 
cols = list(dataset_train)[1:6]
dataset_train = dataset_train[cols]
training_set = dataset_train.as_matrix() # Using multiple predictors.
 
# Feature Scaling
from feature_scaling import feature_scaling
sc = feature_scaling(training_set)[0]
sc_predict = feature_scaling(training_set)[1]
training_set_scaled = feature_scaling(training_set)[2]
 

 
# Creating a data structure with n_past timesteps and n_future outputs
from lstm_data_structure import lstm_data_structure
X_train, y_train = lstm_data_structure(training_set_scaled, 
                                       n_past, 
                                       n_future,
                                       n_features = n_features)
 
# Part 2 - Building the RNN
 
# Import Libraries and packages from Keras
from keras.wrappers.scikit_learn import KerasRegressor
 
# Getting Keras Callbacks. Read Keras callbacks docs for more info.
from get_keras_callbacks import get_keras_callbacks
callbacks = get_keras_callbacks(es_patience = 30,
                                rlr_factor = 0.75,
                                rlr_patience = 5)

# Fitting RNN to training
from build_lstm_regressor import build_lstm_regressor
regressor = KerasRegressor(build_fn = build_lstm_regressor,
                           n_lstm_layers = 6,
                           drop_out = 0.5,
                           n_units = 100,
                           optimizer = 'adam', 
                           n_past = n_past,
                           n_future = n_future)
history = regressor.fit(X_train, 
                        y_train, 
                        shuffle = True, 
                        epochs = 100, 
                        callbacks = callbacks, 
                        validation_split = 0.25, 
                        verbose = 1, 
                        batch_size = 16)
 
 
# Predicting the future.
y_true = np.array(dataset_test.open[len(dataset_test) - n_future:])
 
test_set = dataset_test.iloc[:len(dataset_test) - n_future,1:6].values
test_set_scaled = sc.transform(test_set)

# Creating a data structure with n_past timesteps and n_future outputs
X_test = []
for i in range(0, n_past):
    X_test.append(test_set_scaled[i, 0:n_features])
X_test = np.array(X_test)

# Reshaping
X_test = np.reshape(X_test, (1, X_test.shape[0], X_test.shape[1]))

predictions = regressor.predict(X_test)

y_pred = sc_predict.inverse_transform(predictions).reshape(n_future,1)

# Get MAPE of hold-out predictions
from get_mape import get_mape
get_mape(y_true, y_pred)

hfm, = plt.plot(y_pred, 'r', label='predicted_stock_price')
hfm2, = plt.plot(y_true,'b', label = 'actual_stock_price')
 
plt.legend(handles=[hfm,hfm2])
plt.title('Predictions and Actual Price')
plt.xlabel('Sample index')
plt.ylabel('Stock Price Future')
plt.savefig('graph.png', bbox_inches='tight')
plt.show()
plt.close()
 
# Get MAPE of fitted values
fitted_mape = []
for i in range(0, len(X_train)):
    train_predict = np.reshape(X_train[i], (1, n_past, n_features))
    y_true_train = sc_predict.inverse_transform(y_train[i,:])
    y_fitted = sc_predict.inverse_transform(regressor.predict(train_predict)).reshape(n_future,1)
    fitted_mape.append(get_mape(y_true_train, y_fitted))

np.median(fitted_mape)

hfm, = plt.plot(sc_predict.inverse_transform(y_train[len(y_train)-1,:]), 'r', label='actual_training_stock_price')
hfm2, = plt.plot(sc_predict.inverse_transform(regressor.predict(train_predict)).reshape(n_future,1),'b', label = 'predicted_training_stock_price')
 
plt.legend(handles=[hfm,hfm2])
plt.title('Predictions vs Actual Price')
plt.xlabel('Sample index')
plt.ylabel('Stock Price Training')
plt.savefig('graph_training.png', bbox_inches='tight')
plt.show()
plt.close()