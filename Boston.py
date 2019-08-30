#%% Import Library
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

#%%
seed = 201908
np.random.seed(seed)
tf.set_random_seed(seed)

#%%
df_pre = pd.read_csv('./dataset/housing.csv',
                     header=None,
                     delim_whitespace=True,
                     names=['인구1인당 범죄 발생 수',
                            '25,000m^2 이상 주거구역 비중',
                            '소매업 외 상권크기',
                            '칼스강 변수(1:강주변)',
                            'NOX 농도',
                            '평군 방 개수',
                            '1940이전 지어진 비율',
                            '5가지 보스턴 시 고용시설까지 거리'])

df_pre.info()
df = df_pre.sample(frac=1)
df.head(5)
#%%
dataset = df.values
column_size = dataset.shape[1]-1
X = dataset[:, 0:column_size]
Y = dataset[:, column_size]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.3,
                                                    random_state=seed)

#%%
model = None
model = Sequential()
model.add(Dense(30, input_dim=column_size,activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

#%%
model.compile(loss='mean_squared_error', optimizer='adam')

#%%
model.fit(X_train, Y_train, validation_split=0.33, epochs=1000,
                    batch_size=10, verbose=False)

#%%
Y_prediction = model.predict(X_test).flatten()
for i in range(len(Y_test)):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print("실가격 : %.3f, 예상가격: %.3f" %(label, prediction))

#%%
