#%% Import TF and Set 2-axis Data
import tensorflow as tf
#[공부시간, 과외수업횟수, 성적]
data = [[2,0,81], [4,4,93],[6,2,91],[8,3,97]]
x1_data =[x[0] for x in data]
x2_data = [x[1] for x in data]
y_data =[y[2] for y in data]
print(x1_data)
print(x2_data)
print(y_data)
#%% Set Running Rate
learning_rate = 0.1
#%% Set variable A and b
a1 = tf.Variable(tf.random_uniform([1],0,10, dtype=tf.float64, seed=0))
a2 = tf.Variable(tf.random_uniform([1],0,10,dtype=tf.float64,seed=0))
b = tf.Variable(tf.random_uniform([1],0,100,dtype=tf.float64, seed=0))
print(a1,a2,b)
#%%
y= a1*x1_data + a2*x2_data +b
print(y)
#%% Root Mean Square Error
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))
#%%
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)
#%%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(20001):
        sess.run(gradient_decent)
        if step %100 ==0:
            print("Epoch: %f, RMSE = %.4f, 기울기: a1 = %.4f, 기울기: a2 = %.4f y절편 b = %.4f" %(step,sess.run(rmse),sess.run(a1),sess.run(a2),sess.run(b)))

#%%
