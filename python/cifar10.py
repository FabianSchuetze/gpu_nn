import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import keras
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.callbacks import EarlyStopping
import numpy as np
from keras.datasets import cifar10
DATA = cifar10.load_data()

def get_predictions(model, features):
    """
    Returns predictions and the argpredictions
    """
    predictions = model.predict(features)
    arg_pred = np.argmax(predictions, axis=1)
    return predictions, arg_pred

def prepare_data():
    (x_train, y_train), (x_test, y_test) = DATA
    ohe = OneHotEncoder()
    y_train = ohe.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test = ohe.fit_transform(y_test.reshape(-1, 1)).toarray()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = prepare_data()
    KERAS_SGD = keras.optimizers.SGD(0.001)
    model = keras.Sequential()
    model.add(Conv2D(5, (3, 3), strides=(1, 1), use_bias=False,
              padding='same', input_shape=x_train.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(Conv2D(5, (3, 3), strides=(1, 1), use_bias=False,
              # padding='same'))
    model.add(Activation('relu'))
    # model.add(Conv2D(32, (3, 3)))
    # model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=KERAS_SGD)
    early = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    hist = model.fit(x_train, y_train, epochs=10, verbose=1, batch_size=32,
                     validation_split=0.2, callbacks=[early])
    pred, argpred = get_predictions(model, x_test)
