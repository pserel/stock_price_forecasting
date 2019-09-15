# Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
 
# Importing Data Set
dataset = pd.read_csv('acn.csv')

dataset_train = dataset.iloc[:len(dataset) - 25, :]
dataset_test = dataset.iloc[len(dataset) - 25:, :]
 
cols = list(dataset_train)[1:6]
dataset_train = dataset_train[cols]


 
 
training_set = dataset_train.as_matrix() # Using multiple predictors.
 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
 
sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)
 
sc_predict = StandardScaler()
 
sc_predict.fit_transform(training_set[:,0:1])
 
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
 
n_future = 25  # Number of days you want to predict into the future
n_past = 360  # Number of past days you want to use to predict the future
 
for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past:i, 0:5])
    y_train.append(training_set_scaled[i+n_future-1:i + n_future, 0])
 
X_train, y_train = np.array(X_train), np.array(y_train)
 
# Part 2 - Building the RNN
 
# Import Libraries and packages from Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
 
# Initializing the RNN
regressor = Sequential()
 
# Adding fist LSTM layer and Drop out Regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(n_past, 5)))
regressor.add(Dropout(0.3))
 
# Part 3 - Adding more layers
# Adding 2nd layer with some drop out regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

# Output layer
regressor.add(Dense(units=1, activation='linear'))
 
# Compiling the RNN
regressor.compile(optimizer='adam', loss="mean_squared_error")  # Can change loss to mean-squared-error if you require.
 
# Fitting RNN to training set using Keras Callbacks. Read Keras callbacks docs for more info.
 
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')
 
history = regressor.fit(X_train, y_train, shuffle=True, epochs=100, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=64)
 
 
# Predicting the future.
#--------------------------------------------------------
# The last date for our training set is 05-Aug-2019.
# Lets now try predicting the stocks for the dates in the test set.
 
# The dates on our test set are:
# 6,7,8,9,12,13,14,15,16,19,20,21,22,23,26,27,28,29,30-Aug-2019 and 3,4,5,6,9,10-Sep-2019
 
# Now, the latest we can predict into our test set is to the 19th since the last date on training is 30-Dec-2016. 
# 20 days into the future from the latest day in our training set is 19-Dec-2016. Right?
# Notice that we dont have some days in our test set, what we can do is to take the last 20 samples from the training set. 
# (Remember the last sample of our training set will predict the 19th of Jan 2017, the second last will predict the 18th, etc)
 
 

y_true = np.array(dataset_test['open'])
#Trim the test set to first 12 entries (till the 19th)
y_true = y_true[0:12]
predictions = regressor.predict(X_train[-25:])
 
 
# We skip the 31-Dec, 1-Jan,2-Jan, etc to compare with the test_set
predictions_to_compare = predictions[[3,4,5,6,9,10,11,12,13,17,18,19]]
y_pred = sc_predict.inverse_transform(predictions_to_compare)
 
 
 
hfm, = plt.plot(y_pred, 'r', label='predicted_stock_price')
hfm2, = plt.plot(y_true,'b', label = 'actual_stock_price')
 
plt.legend(handles=[hfm,hfm2])
plt.title('Predictions and Actual Price')
plt.xlabel('Sample index')
plt.ylabel('Stock Price Future')
plt.savefig('graph.png', bbox_inches='tight')
plt.show()
plt.close()
 
 
 
hfm, = plt.plot(sc_predict.inverse_transform(y_train), 'r', label='actual_training_stock_price')
hfm2, = plt.plot(sc_predict.inverse_transform(regressor.predict(X_train)),'b', label = 'predicted_training_stock_price')
 
plt.legend(handles=[hfm,hfm2])
plt.title('Predictions vs Actual Price')
plt.xlabel('Sample index')
plt.ylabel('Stock Price Training')
plt.savefig('graph_training.png', bbox_inches='tight')
plt.show()
plt.close()