#%% Import pandas, seaborn, matplolib, sklearn, keras
import pandas as pd
import seaborn as sns
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
file_path = 'dataset/iris.csv'

#%% Read data
df = pd.read_csv(file_path, names=[
                 "sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
df.head(5)
#%% draw as pairplot
sns.pairplot(df, hue="species")
plt.show()

#%% Using One-Hot-Encoding (Cause Species is String. So We need to Convert Number)
dataset = df.values
X = dataset[:, 0:4].astype(float)
Y_obj = dataset[:, 4]
print(Y_obj.reshape([len(Y_obj), 1]))

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = np_utils.to_categorical(Y)
print(Y_encoded)
#%% define model (Using relu and softmax. Softmax is Sum must be 1)
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))


#%% Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


#%% Run model
model.fit(X, Y_encoded, epochs=100, batch_size=1)


#%%
print('Accuracy" %.4f' % (model.evaluate(X, Y_encoded)[1]))

#%%
