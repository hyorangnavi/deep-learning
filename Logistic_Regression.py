#%% import numpy and tf
import numpy as np
import tensorflow as tf
#%%
data = [[2,0], [4,0],[6,0],[8,1],[10,1],[12,1],[14,1]]
x_data = [i[0] for i in data]
y_data = [i[1] for i in data]
#%%
a = tf.Variable(tf.random_uniform([1],dtype=tf.float64,seed=0))
b = tf.Variable(tf.random_uniform([1],dtype=tf.float64,seed=0))