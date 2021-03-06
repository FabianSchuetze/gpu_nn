from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
import os

batch_size = 32
num_classes = 10
epochs = 3


def normalize(data):
    std = data.std(axis=0, keepdims=True)
    mean = data.mean(axis=0, keepdims=True)
    return (data - mean) / std

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# CNN architecture with Keras
model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same',
                 input_shape=x_train.shape[1:],
                 kernel_regularizer=l2(0.000)))
# model.add(Conv2D(input_shape=trainX[0,:,:,:].shape, filters=32,
                 # use_bias=True, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Conv2D(filters=64, use_bias=False, kernel_size=(5,5), strides=2))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation="softmax"))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=    ['accuracy'])

# model = Sequential()
# model.add(Conv2D(32, (5, 5), padding='same',
                 # input_shape=x_train.shape[1:],
                 # kernel_regularizer=l2(0.000)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(32, (5, 5), padding='same', kernel_regularizer=l2(0.000)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(64, (5, 5), padding='same', kernel_regularizer=l2(0.000)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Flatten())
# model.add(Dense(64, kernel_regularizer=l2(0.000)))
# model.add(Activation('relu'))
# model.add(Dense(num_classes, kernel_regularizer=l2(0.000)))
# model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

hist = model.fit(normalize(x_train), y_train, batch_size=batch_size, 
                 epochs=epochs, validation_split=0.2)
