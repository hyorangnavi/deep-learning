#%%
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
#%%
seed = 201908
np.random.seed(seed)
tf.set_random_seed(seed)
model = None

#%%
df_pre = pd.read_csv('dataset/Wine.csv', header=None, names=["주석산농도", "아세트산농도", "구현산농도", "잔류 당분 농도",
                                                             "염화나트륨 농도", "유리 아황산 농도", " 총 아황산 농도", "밀도", "pH", "황산칼륨 농도", "알콜 도수", "와인맛(0~10)", "Class"])
df_pre.info()

df = df_pre.sample(frac=0.15)
df.info()
#%%
dataset = df.values
column_size = dataset.shape[1]-1
X = dataset[:, 0:column_size]
Y = dataset[:, column_size]
#%%
model = Sequential()
model.add(Dense(30, input_dim=column_size, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#%%
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

#%%
MODEL_DIR = './model/Wine/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

#modelpath = MODEL_DIR + "{epoch:02d}-{val_loss:.4f}.hdf5"
modelpath = MODEL_DIR + "best_model.hdf5"
#%% patience is like Yellow Card
checkpointer_callback = ModelCheckpoint(
    filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100) #if model doesn't improve next prev 100 times, then stop
#%% 
history = model.fit(X, Y, validation_split=0.33, epochs=3500,
                    batch_size=500, verbose=False, callbacks=[checkpointer_callback, early_stopping_callback])

#%% 
y_vloss = history.history['val_loss']
y_acc = history.history['acc']

x_len = np.arange(len(y_acc))
plt.plot(x_len,y_vloss,"o",c="red",markersize=3)
plt.plot(x_len,y_acc,"o",c="blue",markersize=3)
plt.show()

#%%
if model is not None:
    del model
model = load_model(modelpath)

#%%
print('Loaded Model\'s Accuracy: %.4f' %(model.evaluate(X,Y)[1]))

#%%
