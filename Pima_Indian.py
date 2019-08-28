#%% import pandas, matplotlib and seaborn
import tensorflow as tf
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_path = 'dataset/pima-indians-diabetes.csv'
#%% read 'pima-indians-diabetes.csv' data
df = pd.read_csv(file_path,
                 names=["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])
#%% print 0 to 4 rows
print(df.head(5))
#%% print information about df
# brief Info
print(df.info())
# More Info
print(df.describe())

# Some Info
print(df[['pregnant', 'class']])
#%% Find Relationship between Pregnant Count and Class , and draw it
print(df[['pregnant', 'class']]
      .groupby(['pregnant'], as_index=False)
      .mean()
      .sort_values(by="pregnant", ascending=True))

#%% defint plt.figure's size as 864 864
plt.figure(figsize=(12, 12))
#%% without heatmap, figure-out df's Correlation
print(df.corr())
#%% draw heatmap
sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5,
            cmap=plt.cm.gist_heat, linecolor='white', annot=True)
plt.show()
#%% draw FacetGrid
grid = sns.FacetGrid(df, col='class')
grid.map(plt.hist, 'plasma', bins=10)

#%% Then now, using Keras and Predict diabetes result
#%% First You need to import Keras, numpy and Tensorflow

#%% set Random Seed
seed = 201809
np.random.seed(seed)
tf.set_random_seed(seed)

#%% read data and cut as ',' set x as variables y as class
dataset = np.loadtxt(file_path, delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

#%% set model input->Dense12(relu)->Dense8(relu)->output(Sigmoid)
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # <- Output

#%% Compile model (Loss Function  = Binary CrossEntropy)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#%% Run model
model.fit(X,Y, epochs = 1000, batch_size=10)

#%%
print("Accuracy is : %.4f" % (model.evaluate(X,Y)[1]))

#%%
