#%%
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

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

#%% Split 10 files
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold,shuffle=True,random_state=seed)

accuracy= []
#%%
for train, test in skf.split(X,Y):
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(X[train],Y[train],epochs=100,batch_size=5)
    accuracy.append('%.4f' % (model.evaluate(X[test], Y[test])[1]))

#%%
print(" %d fold Accuracy:" %(n_fold), accuracy)


#%%
