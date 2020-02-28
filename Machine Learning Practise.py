# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:32:27 2020

@author: Careen
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##Importing the data set
car_df= pd.read_csv("autos_small.csv")
car_df


## Visualize the data
sns.pairplot(car_df)

##CLEANING THE DATA
X = car_df.drop(['name','model', 'fuelType','price'], axis=1)

X

y= car_df['price']

y


##Scaling the data set
from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler()
X_scaled= scaler.fit_transform(X)
X_scaled

X_scaled.shape

scaler.data_max_

scaler.data_min_

y= y.values.reshape(-1,1)
y_scaled= scaler.fit_transform(y)
y_scaled

y_scaled.shape

##Training the Model
##We use train-test to split the data

from sklearn.model_selection import train_test_split

##25% is the size of testing data set in this model 
##If 25% is the testing data then 75% is the training data

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size= 0.25) 

X_train.shape

X_test.shape

##Building the ANN in layers

import tensorflow.keras #keras is an API
from keras.models import Sequential #Sequential means the model is built in sequential form
from keras.layers import Dense

model= Sequential()
model.add(Dense(40,input_dim =3, activation = 'relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(1, activation = 'linear')) # the 1 represents the output y which is one


model.summary()#for showing how many neurons and chainable parameters are there


model.compile(optimizer = 'adam', loss= 'mean_squared_error')

epochs_hist =model.fit(X_train, y_train, epochs= 100, batch_size= 50, verbose=1, validation_split= 0.2)

## Evaluating the Model

epochs_hist.history.keys()

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.ylabel('Training and Validation Loss')
plt.xlabel('Epochs number')
plt.legend(['Training Loss', 'Validation Loss'])

#YearofRegression, PowerPs, Kilometer

X_test = np.array([[2002, 150,4000]])
y_predict= model.predict(X_test)

print('price', y_predict)