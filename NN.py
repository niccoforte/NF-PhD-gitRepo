import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer, BatchNormalization, Activation, Dropout, LSTM

inputs = [1,1,1,1,1,1]
outputs = [2,2]


model = Sequential(name='FCNN')

model.add(InputLayer(input_shape=len(inputs)))
model.add(BatchNormalization())

model.add(Dense(2048))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(LSTM(1024))
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(LSTM(512))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(len(outputs)))