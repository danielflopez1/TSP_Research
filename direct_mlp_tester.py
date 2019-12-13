'''
Using this file to test multilple MLP activation combinations
'''
from keras import initializers
from keras.activations import tanh, softmax
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
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
        self.xs = np.array(self.xs)/100
        print(self.xs,self.xs.shape)

        pickle_file = open('ys.pickle', 'rb')
        self.ys = pickle.load(pickle_file)
        pickle_file.close()
        self.ys = np.array(self.ys)/10


    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.xs, self.ys, test_size = 0.2, random_state = 42)
        print(X_train.shape)
        model = Sequential()
        model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu')) #Mutli layer perceptron inputs
        model.add(Dropout(0.2))
        model.add(Dense(240, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(480, activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(480, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(240, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(9, activation='relu'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # fit the keras model on the dataset
        model.fit(X_train, y_train, epochs=150, batch_size=20, verbose=1,validation_data=(X_test,y_test))
        # make class predictions with the model
        predictions = model.predict(X_train[:10])
        for i, pred in enumerate(predictions):
            print(predictions[i], "==", y_train[i])


if __name__=="__main__":
    dnn = NNTSP()
    dnn.get_pickles()
    dnn.train()