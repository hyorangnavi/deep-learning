#%% Import Package and Library
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import tensorflow as tf

#%% Set Random Seed
seed = 201809
np.random.seed(seed)
tf.set_random_seed(seed)

#%% Load Dataset
Data_set = np.loadtxt("dataset/ThoraricSurgery.csv", delimiter=",")

#%% X = Surgery Info, Y = Result
X = Data_set[:, 0:17]
Y = Data_set[:, 17]

#%% make Deep-Learning Model (Densely-NN Connected Layer,30-Nodes, 17-Inputs, act.Func-Relu)
model = Sequential()
model.add(Dense(30,input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#%% Run Deep-Learning (loss-func : binary_crossentropy)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y, epochs=40, batch_size=10)

#%%
print("\n Accuracy: %.4f" %(model.evaluate(X,Y)[1]))

#%%
