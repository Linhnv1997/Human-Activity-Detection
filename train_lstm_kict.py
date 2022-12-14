import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

from sklearn.model_selection import train_test_split

# Đọc dữ liệu
standing_df = pd.read_csv("./Data/Standing.txt")
eating_df = pd.read_csv("./Data/Eating.txt")
watching_df = pd.read_csv("./Data/Watching_TV.txt")
# clapp_df = pd.read_csv("Clapping.txt")
talking_df = pd.read_csv("./Data/talking.txt")

X = []
y = []
no_of_timesteps = 20


dataset = standing_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(np.array([1., 0., 0.,0.]))

dataset = eating_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(np.array([0., 1., 0.,0.]))

dataset = watching_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(np.array([0., 0., 1.,0.]))

dataset = talking_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(np.array([0., 0.,0.,1.]))

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)
# print(X)
# columnTransformer = ColumnTransformer([('encoder',
#                                         OneHotEncoder(),
#                                         [0])],
#                                       remainder='passthrough')
# onehotencoder = OneHotEncoder()
# y = onehotencoder.fit_transform(y)
#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units =4, activation="softmax"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "categorical_crossentropy")

model.fit(X_train, y_train, epochs=20, batch_size=64,validation_data=(X_test, y_test))
model.save("modellstm.h5")


