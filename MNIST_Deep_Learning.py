#%%
from keras.datasets import mnist
import matplotlib.pyplot as plt
#%%
(X_train, Y_class_train), (X_test, Y_class_train) = mnist.load_data()

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

X_train[test_num]
print("Total Cell size is: ", len(X_train[test_num])*len(X_train[test_num][0]))

#%%


#%%
