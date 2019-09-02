#%%
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
#%%
(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data()

#%%
print("학습된 이미지 수 : {:d}개".format(X_train.shape[0]))
print("테스트셋 이미지 수 : {:d}개".format(X_test.shape[0]))

#%%
test_num =4
plt.imshow(X_train[test_num], cmap='Greys')
plt.show()

for x in X_train[test_num]:
    for i in x:
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')

cell_size = len(X_train[test_num])*len(X_train[test_num][0])
print("Total Cell size is: ",cell_size)

#%%
X_train = X_train.reshape(X_train.shape[0],-1).astype('float64') / 255
#%%
X_test = X_test.reshape(X_test.shape[0],-1).astype('float64') / 255

#%%
print("class : {:d}".format(Y_class_train[test_num]))
Y_class_train = np_utils.to_categorical(Y_class_train,10)
Y_class_test = np_utils.to_categorical(Y_class_test,10)

#%%
print(Y_class_train[test_num])
print(Y_class_test[test_num])

#%%
model = Sequ