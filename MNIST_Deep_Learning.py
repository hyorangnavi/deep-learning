#%%
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
#%%
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#%%
print("학습된 이미지 수 : {:d}개".format(X_train.shape[0]))
print("테스트셋 이미지 수 : {:d}개".format(X_test.shape[0]))

#%%
test_num = 4
plt.imshow(X_train[test_num], cmap='Greys')
plt.show()

for x in X_train[test_num]:
    for i in x:
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')

cell_size = len(X_train[test_num])*len(X_train[test_num][0])
print("Total Cell size is: ", cell_size)

#%%
X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
#%%
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255
#%%
X_train[test_num]
#%%
print("class : {:d}".format(Y_train[test_num]))
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

#%%
print(Y_train[test_num])
print(Y_test[test_num])

#%%
model = Sequential()
model.add(Dense(512, input_dim=cell_size, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

#%%
MODEL_DIR = './model/MNIST/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

#%%
modelpath = MODEL_DIR+'best_model.hdf5'
checkpoint_callback = ModelCheckpoint(
    filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

#%%
history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpoint_callback])

#%%
print('Test Accuracy : %.4f' % (model.evaluate(X_test,Y_test)[1]))
#%%
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

#%%
x_len = np.arange(len(y_loss))
plt.plot(x_len,y_vloss, '-',c='red',label='Testset_loss')
plt.plot(x_len,y_loss,'-',c='blue',label='Trainset_loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#%%
