from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping