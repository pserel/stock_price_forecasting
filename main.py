# Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
 
# Importing Data Set
dataset = pd.read_csv('acn.csv')

n_future = 25  # Number of days you want to predict into the future
n_past = 360  # Number of past days you want to use to predict the future
dataset_train = dataset.iloc[:len(dataset) - n_future - n_past, :]
dataset_test = dataset.iloc[len(dataset) - n_future - n_past:, :]
 
cols = list(dataset_train)[1:6]
dataset_train = dataset_train[cols]


 
 
training_set = dataset_train.as_matrix() # Using multiple predictors.
 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
 
sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)
 
sc_predict = StandardScaler()
 
sc_predict.fit_transform(training_set[:,0:1])
 
# Creating a data structure with 360 timesteps and 25 outputs
X_train = []
y_train = []
 
 
for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past:i, 0:5])
    #y_train.append(training_set_scaled[i+n_future-1:i + n_future, 0])
    y_train.append(training_set_scaled[i:i + n_future, 0])
 
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
regressor.add(Dropout(0.4))
 
# Part 3 - Adding more layers
# Adding 2nd layer with some drop out regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.4))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.4))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.4))

# Output layer
regressor.add(Dense(units=n_future))
 
# Compiling the RNN
regressor.compile(optimizer='adam', loss="mean_squared_error")  # Can change loss to mean-squared-error if you require.
 
# Fitting RNN to training set using Keras Callbacks. Read Keras callbacks docs for more info.
 
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')
 
history = regressor.fit(X_train, y_train, shuffle=True, epochs=100, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=128)
 
 
# Predicting the future.
 

y_true = np.array(dataset_test.open[len(dataset_test) - n_future:])
 
test_set = dataset_test.iloc[:len(dataset_test) - n_future,1:6].values
test_set_scaled = sc.transform(test_set)

# Creating a data structure with 360 timesteps and 25 outputs
X_test = []
for i in range(0, n_past):
    X_test.append(test_set_scaled[i, 0:5])
X_test = np.array(X_test)

# Reshaping
X_test = np.reshape(X_test, (1, X_test.shape[0], X_test.shape[1]))

predictions = regressor.predict(X_test)

y_pred = sc_predict.inverse_transform(predictions).reshape(25,1)

hfm, = plt.plot(y_pred, 'r', label='predicted_stock_price')
hfm2, = plt.plot(y_true,'b', label = 'actual_stock_price')
 
plt.legend(handles=[hfm,hfm2])
plt.title('Predictions and Actual Price')
plt.xlabel('Sample index')
plt.ylabel('Stock Price Future')
plt.savefig('graph.png', bbox_inches='tight')
plt.show()
plt.close()
 
 
train_predict = np.reshape(X_train[len(X_train)-1], (1, 360, 5))
hfm, = plt.plot(sc_predict.inverse_transform(y_train[len(y_train)-1,:]), 'r', label='actual_training_stock_price')
hfm2, = plt.plot(sc_predict.inverse_transform(regressor.predict(train_predict)).reshape(25,1),'b', label = 'predicted_training_stock_price')
 
plt.legend(handles=[hfm,hfm2])
plt.title('Predictions vs Actual Price')
plt.xlabel('Sample index')
plt.ylabel('Stock Price Training')
plt.savefig('graph_training.png', bbox_inches='tight')
plt.show()
plt.close()