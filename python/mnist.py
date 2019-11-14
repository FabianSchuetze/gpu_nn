"""
Tests different neural networks
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import keras
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import NeuralNetwork as ex
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.close("all")
from keras.datasets import mnist
DATA = mnist.load_data()

def prepare_data():
    """
    Returns the mnist data
    """
    scaler = StandardScaler()
    ohe = OneHotEncoder()
    # The digits dataset
    digits = datasets.load_digits()
    features = scaler.fit_transform(digits['data'])
    target = ohe.fit_transform(digits['target'].reshape(-1, 1)).toarray()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2)
    return x_train, x_test, y_train, y_test


# def plot_losses(axis, keras_history, my_history):
    # """
    # Plots all validation losses during training
    # """
    # keras_val_loss = keras_history.history["val_loss"]
    # my_val_loss = my_history.get_losses()["validation_loss"]
    # axis.plot(np.arange(len(keras_val_loss)), keras_val_loss,
              # color="red", label="keras")
    # axis.plot(np.arange(len(my_val_loss)), my_val_loss,
              # color="blue", label="mine")


# def get_predictions(model, features):
    # """
    # Returns predictions and the argpredictions
    # """
    # predictions = model.predict(features)
    # arg_pred = np.argmax(predictions, axis=1)
    # return predictions, arg_pred


# def compare_missclassified(keras_argpred, my_argpred, actual):
    # """
    # Calculates the number of missclassified samples
    # """
    # argtarget = np.argwhere(actual)[:, 1]
    # missclassified_mine = ((my_argpred - argtarget) != 0).sum()
    # missclassified_keras = ((keras_argpred - argtarget) != 0).sum()
    # print("N missclassied (mine) %i" % (missclassified_mine))
    # print("N missclassied (keras) %i" % (missclassified_keras))
    # return missclassified_keras, missclassified_mine


# def return_my_predictions(architecture, optimizer, samples):
    # x_train, x_test, y_train, _ = samples
    # model = ex.NeuralNetwork(architecture, "Categorical_Crossentropy")
    # hist = model.train(x_train, y_train, 100, optimizer, 32, 0.2, 20)
    # pred, argpred = get_predictions(model, x_test)
    # return hist, pred, argpred


# def return_keras_predictions(optimizer, samples):
    # x_train, x_test, y_train, _ = samples
    # model = keras.Sequential()
    # model.add(Dropout(0.2))
    # model.add(Dense(512, input_shape=(x_train.shape[1],), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    # # model.add(BatchNormalization())
    # model.add(Dense(10, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    # early = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    # hist = model.fit(x_train, y_train, epochs=50, verbose=1, batch_size=32,
                     # validation_split=0.2, callbacks=[early])
    # pred, argpred = get_predictions(model, x_test)
    # return hist, pred, argpred

def test_mine_SGD():
    # sgd = ex.StochasticGradientDescent(0.001, 0.)
    return nn


# def test_mine_Momentum():
    # architecture = []
    # architecture.append(ex.Input(64))
    # architecture.append(ex.Dense(20, True))
    # architecture.append(ex.Activation("relu"))
    # architecture.append(ex.Dropout(0.5))
    # architecture.append(ex.Dense(10, True))
    # architecture.append(ex.Activation("softmax"))
    # sgd = ex.Momentum(0.001, 0.5, architecture, 0.)
    # argpred = return_my_predictions(architecture, sgd, SAMPLES)[2]
    # argtarget = np.argwhere(SAMPLES[3])[:, 1]
    # missclassified_mine = ((argpred - argtarget) != 0).sum()
    # print("missclassified_mine: %s" % (missclassified_mine))
    # assert missclassified_mine < 70, "too much error"


# def test_mine_RMSProp():
    # architecture = []
    # architecture.append(ex.Input(64))
    # architecture.append(ex.Dense(20, True))
    # architecture.append(ex.Activation("relu"))
    # architecture.append(ex.Dropout(0.5))
    # architecture.append(ex.Dense(10, True))
    # architecture.append(ex.Activation("softmax"))
    # sgd = ex.RMSProp(0.001, 0.9, architecture)
    # argpred = return_my_predictions(architecture, sgd, SAMPLES)[2]
    # argtarget = np.argwhere(SAMPLES[3])[:, 1]
    # missclassified_mine = ((argpred - argtarget) != 0).sum()
    # assert missclassified_mine < 50, "too much error"


# def test_mine_Adam():
    # architecture = []
    # architecture.append(ex.Input(64))
    # architecture.append(ex.Dense(20, True))
    # architecture.append(ex.Activation("relu"))
    # architecture.append(ex.Dropout(0.5))
    # architecture.append(ex.Dense(10, True))
    # architecture.append(ex.Activation("softmax"))
    # sgd = ex.Adam(0.001, 0.9, 0.999, architecture)
    # argpred = return_my_predictions(architecture, sgd, SAMPLES)[2]
    # argtarget = np.argwhere(SAMPLES[3])[:, 1]
    # missclassified_mine = ((argpred - argtarget) != 0).sum()
    # assert missclassified_mine < 50, "too much error"


if __name__ == "__main__":
# x_train, x_test, y_train, y_test = prepare_data()
    x_train, x_test, y_train, y_test = prepare_data()
    SAMPLES = (x_train,x_test, y_train,y_test)
    sgd = ex.StochasticGradientDescent(0.001)
    loss = ex.CrossEntropy()
    architecture = []
    architecture.append(ex.Input(64))
    architecture.append(ex.Dense(20, 64))
    architecture.append(ex.Relu())
    architecture.append(ex.Dense(10, 20))
    architecture.append(ex.Softmax())
    loss = ex.CrossEntropy()
    nn = ex.NeuralNetwork(architecture, loss, "CPU")
    # NN = test_mine_SGD()
    nn.train(x_train, y_train, sgd)
