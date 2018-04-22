# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 2018

@author: Zeynep CANKARA
"""
"import useful libraries"

"inspired by Udemy Course DeepLearning A-Z"

# data analysis
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt



"Stock Market Prediction on Turkish Stock Index BIST"
dataset_train = pd.read_csv('BIST100.csv')

"stock price predictor only will trained according to opening price"

"replace empty opening prices with the median"
dataset_train['Open'].fillna(dataset_train['Open'].dropna().median(), inplace=True)

"take your training set"
x_train = dataset_train.iloc[:, 1:2].values

"apply feture scaling"
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 10))
training_set_scaled = sc.fit_transform(x_train)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 734):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN
"Sequential model LSTML with dropout regularisation"
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

"end neural network model"
# Getting the real stock price of 2018
dataset_test = pd.read_csv('BIST100_test.csv')
"get the opening price of stocks Oct 22, 2017 - Apr 22, 2018"
real_stock_price = dataset_test.iloc[:, 1:2].values

"prepare the training set"
# Getting the predicted stock price of 2017
dataset_total = dataset_train['Open']
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 141):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualise the graph
plt.plot(real_stock_price, color = 'red', label = 'Real BIST100 Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted BIST100 Price')
plt.title('BIST100 Prediction 2018')
plt.xlabel('Time: Oct 22, 2017 - Apr 22, 2018 ')
plt.ylabel('BIST100 Price')
plt.legend()
plt.show()
