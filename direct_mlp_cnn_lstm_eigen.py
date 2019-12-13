
from keras import initializers
from keras.activations import tanh, softmax
import keras.backend as K
from keras.layers import Conv2D, Flatten,MaxPool2D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy.linalg as LA
from  keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

class NNTSP:
    def __init__(self):
        self.xs = []
        self.ys = []
        self.eigens = []
    def get_pickles(self,eigs = True):
        pickle_file = open('xs.pickle', 'rb')
        self.xs = pickle.load(pickle_file)
        pickle_file.close()
        self.xs = np.array(self.xs)
        if(eigs):                   #If Eigen is true, set the eigen values prepared
            self.xs = self.xs.reshape((100000, 10, 10))
            for i in range(len(self.xs)):
                self.eigens.append(LA.eig(self.xs[0])[1])
            self.xs= np.array(self.eigens).reshape((100000, 10, 10,1))
        pickle_file = open('ys.pickle', 'rb')
        self.ys = pickle.load(pickle_file)
        pickle_file.close()
        self.ys = np.array(self.ys)/10

    #Train and test MLP accuracy with both adjacency matrix and adjacency eigenvalue matrix
    def mlp_tsp_train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.xs, self.ys, test_size = 0.2, random_state = 42) #split the training and testing data
        print(X_train.shape)
        model = Sequential()
        model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu')) #Mutli layer perceptron inputs
        model.add(Dropout(0.2))                                                         #Use dropout to reduce overfitting
        model.add(Dense(240, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(480, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(480, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(240, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(9, activation='sigmoid'))                                       #Finalize with sigmoid
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.summary()
        # fit the keras model on the dataset
        earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.000001, #End training early if there is no modification
                                      patience=20, verbose=1, mode='auto')
        history  = model.fit(X_train, y_train, epochs=150, batch_size=20, verbose=1,callbacks=[earlyStopping],validation_data=(X_test,y_test)) #Start training

        # make class predictions with the model
        model.save("MLP.model")
        predictions = model.predict(X_test[:10]) #Create some predictions

        for i, pred in enumerate(predictions):      #check the predictions
            print(predictions[i], "==", y_test[i])

    # Train and test CNN accuracy with both adjacency matrix and adjacency eigenvalue matrix
    def cnn_tsp_train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.xs, self.ys, test_size = 0.2, random_state = 42)
        print(X_train.shape)
        model = Sequential()
        model.add(Conv2D(100, (3, 3), strides=1, input_shape=(10, 10, 1)))   #Convolutional network input
        model.add(MaxPool2D((2, 2)))                                         #Maxpoool
        model.add(Conv2D(100, (2, 2)))
        model.add(MaxPool2D((2, 2)))
        model.flatten()
        model.add(Dropout(0.2))
        model.add(Dense(240, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(480, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(480, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(240, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(9, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.summary()
        # fit the keras model on the dataset
        earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.000001,
                                      patience=20, verbose=1, mode='auto')
        history  = model.fit(X_train, y_train, epochs=150, batch_size=20, verbose=1,callbacks=[earlyStopping],validation_data=(X_test,y_test))

        # make class predictions with the model
        model.save("CNN.model")
        predictions = model.predict(X_test[:10])

        for i, pred in enumerate(predictions):
            print(predictions[i], "==", y_test[i])

    # Train and test LSTM accuracy with both adjacency matrix and adjacency eigenvalue matrix
    def lstm_tsp_train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.xs, self.xs, test_size = 0.2, random_state = 42)
        print(X_train.shape)
        model = Sequential()
        model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'))
        model.flatten()
        model.add(LSTM(100,return_sequences=True))
        model.add(MaxPool2D((2,2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(240, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.summary()
        # fit the keras model on the dataset
        earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.000001,
                                      patience=10, verbose=1, mode='auto')
        history  = model.fit(X_train, y_train, epochs=10, batch_size=20, verbose=1,callbacks=[earlyStopping],validation_data=(X_test,y_test))

        # make class predictions with the model
        model.save("LSTM.model")
        predictions = model.predict(X_test[:10])

        for i, pred in enumerate(predictions):
            print(predictions[i], "==", y_test[i])



if __name__=="__main__":
    dnn = NNTSP()
    dnn.get_pickles()
    dnn.mlp_tsp_train()
    dnn.cnn_tsp_train()
    dnn.lstm_tsp_train()