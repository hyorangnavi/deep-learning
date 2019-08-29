#%%
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder


file_path = 'dataset/sonar.csv'

#%%
df = pd.read_csv(file_path,header=None)
df
df.info()
df.head(5)
#%%
