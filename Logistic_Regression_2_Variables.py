#%% Import tensorflow and numpy
import tensorflow as tf
import numpy as np
#%% Generate Random seeds
seed = 201908
np.random.seed(seed)
tf.set_random_seed(seed)
#%% Set Data : X,Y
x_data = np.array([[2,3],[4,3],[6,4],[8,6],[10,7],[12,8],[14,9]])
y_data = np.array([0,0,0,1,1,1,1]).reshape(7,1)
print(x_data)
print(y_data)
#%% Save input Variables to placeHolder
X = tf.placeholder(tf.float64, shape=[None,2])
Y = tf.placeholder(tf.float64, shape=[None,1])

#%% Set random Slope a and bias b
a = tf.Variable(tf.random_uniform([2,1], dtype=tf.float64))
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))

#%% Set Sigmoid Func
y = tf.sigmoid(tf.matmul(X,a) +b)
loss = -tf.reduce_mean(Y * tf.log(y) + (1-Y)* tf.log(1-y))
#%%
learning_rate = 0.1
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#%%
predicted = tf.cast(y>0.5, dtype=tf.float64)
accuracy = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#%%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3001):
        a_, b_, loss_, _ = sess.run([a, b, loss,gradient_decent], feed_dict={X: x_data, Y: y_data})
        if i%100 ==0:
            print("Epoch: %d, a1= %.4f, a2= %.4f, b== %.4f, loss= %.4f" %(i,a_[0],a_[1],b_,loss_))
