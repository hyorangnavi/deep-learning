#%% import numpy and tf
import numpy as np
import tensorflow as tf
#%%
data = [[2,0], [4,0],[6,0],[8,1],[10,1],[12,1],[14,1]]
x_data = [i[0] for i in data]
y_data = [i[1] for i in data]
print(x_data)
print(y_data)
#%%
a = tf.Variable(tf.random_normal([1],dtype=tf.float64,seed=0))
b = tf.Variable(tf.random_normal([1],dtype=tf.float64,seed=0))
print(a,b)
#%% define y-Sigmoid func
y = 1/(1+np.e**(-a*x_data + b))
#%% defin loss func
loss = -tf.reduce_mean(np.array(y_data)*tf.log(y) + (1-np.array(y_data))*tf.log(1-y))

#%%
learning_rate = 0.5
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#%%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(60001):
        sess.run(gradient_decent)
        if i % 6000 == 0:
            print("Epoch: %.f, loss= %.4f, 기울기 a= %.4f, y절편 = %.4f" %(i,sess.run(loss), sess.run(a), sess.run(b)))

#%%
