#%%
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.utils import np_utils
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy

# %%
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

# %%
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)

# %%
