#%%
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

file_path = 'dataset/sonar.csv'
#%%
df = pd.read_csv(file_path, header=None)
df
df.info()
df.head(5)
#%% Set Random Seed
seed = 201908
tf.set_random_seed(seed)
np.random.seed(seed)

#%%
dataset = df.values
X = dataset[:,0:df.shape[1]-1]
Y_obj = dataset[:,df.shape[1]-1]

#%%
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
print(Y)

#%% Divide into Test-Set and Train-Set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state= seed)

#%%
model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#%%
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=200,batch_size=5)

#%%
print("Accuracy: %.4f" %( model.evaluate(X_test,Y_test)[1]))

#%% save model
model.save('Sonar_model.h5')

#%%
del model
model = load_model('Sonar_model.h5')

#%%
print('Loaded Model\'s Accuracy: %.4f' %(model.evaluate(X_test,Y_test)[1]))

#%%
