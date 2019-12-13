'''
Using this file to test different loss functions
'''

from keras import initializers
from keras.activations import tanh, softmax
import keras.backend as K
from keras.layers import Conv2D, Flatten,MaxPool2D,Add, Input, Conv1D, LSTM
from keras.models import Sequential, Model
from keras.layers import Dense,Concatenate
from keras.layers import Dropout
from  keras.callbacks import EarlyStopping
import tensorflow as tf
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

class NNTSP:
    def __init__(self):
        self.xs = []
        self.ys = []

    def get_pickles(self):
        pickle_file = open('xs.pickle', 'rb')
        self.xs = pickle.load(pickle_file)
        pickle_file.close()
        self.xs = np.array(self.xs).reshape((10000, 10, 10, 1)) / 100

        pickle_file = open('ys.pickle', 'rb')
        self.ys = pickle.load(pickle_file)
        pickle_file.close()
        self.ys = np.array(self.ys) / 10 - 0.1

    # Train and test MLP accuracy with both adjacency matrix and adjacency eigenvalue matrix
    def mlp_loss_tsp_train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.xs, self.ys, test_size=0.2, random_state=42)
        inputs = Input(shape=(10,10,1))
        flat = Flatten()(inputs)
        layer1 = Dense(200, activation='relu')(flat)
        out = Dense(9, activation='sigmoid')(layer1)
        model = Model(inputs=inputs, outputs=out)

        def mod_loss(y_true,y_pred):  #Change loss function to test given tsp can have reversed paths in undirected adjacency matrix
            tf.print(y_true)
            return tf.minimum(K.mean(K.square(y_pred - tf.reverse(y_true,-1)), axis=-1),K.mean(K.square(y_pred - y_true), axis=-1))

        model.compile(loss=mod_loss, optimizer='adam', metrics=['accuracy'])
        model.summary()

        # fit the keras model on the dataset
        history = model.fit(X_train, y_train, epochs=1, batch_size=20, verbose=1, callbacks=[WandbCallback()],
                            validation_data=(X_test, y_test),)

        # make class predictions with the model
        model.save("NNTSP.model")
        predictions = model.predict(X_train[:10])

        for i, pred in enumerate(predictions):
            print(predictions[i], "==", y_train[i])
