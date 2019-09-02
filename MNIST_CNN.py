#%%
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
#%%
seed = 201809
np.random.seed(seed)
tf.set_random_seed(seed)

#%%
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#%%
print('X_train[0] shape is :', X_train[0].shape)

#%%
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
#%%
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

#%%
MODEL_DIR = './model/MNIST_CNN/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

#%%
modelpath = MODEL_DIR+'best_model.hdf5'
checkpoint_callback = ModelCheckpoint(
    filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)


#%%
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpoint_callback])

#%%
