from keras import initializers
from keras.activations import tanh, softmax
import keras.backend as K
from keras.layers import Conv2D, Flatten,MaxPool2D,Add, Input, Conv1D, LSTM
from keras.models import Sequential, Model
from keras.layers import Dense,Concatenate
from keras.layers import Dropout
from  keras.callbacks import EarlyStopping

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
        self.xs = np.array(self.xs).reshape((10000,10,10,1)) / 100
        #print(self.xs,self.xs.shape)

        pickle_file = open('ys.pickle', 'rb')
        self.ys = pickle.load(pickle_file)
        pickle_file.close()
        self.ys = np.array(self.ys)/10 -0.1

    def cnn_path_train(self):
        dout = 0
        X_train, X_test, y_train, y_test = train_test_split(self.xs, self.ys, test_size = 0.2, random_state = 42)  #Split the data

        #Generate adjacency matrix and append it to the vectors. Each vector will be appended to its corresponding adjacency matrix.
        r0 = []
        t0 = []
        r1 = []
        t1 = []
        r2 = []
        t2 = []
        r3 = []
        t3 = []
        r4 = []
        t4 = []
        r5 = []
        t5 = []
        r6 = []
        t6 = []
        r7 = []
        t7 = []
        r8 = []
        t8 = []
        r9 = []
        t9 = []

        for matrix in X_train:  #Append
            r0.append(np.append(matrix[0].T,matrix[:,0].T))
            r1.append(np.append(matrix[1].T, matrix[:, 1].T))
            r2.append(np.append(matrix[2].T, matrix[:, 2].T))
            r3.append(np.append(matrix[3].T, matrix[:, 3].T))
            r4.append(np.append(matrix[4].T, matrix[:, 4].T))
            r5.append(np.append(matrix[5].T, matrix[:, 5].T))
            r6.append(np.append(matrix[6].T, matrix[:, 6].T))
            r7.append(np.append(matrix[7].T, matrix[:, 7].T))
            r8.append(np.append(matrix[8].T, matrix[:, 8].T))
            r9.append(np.append(matrix[9].T, matrix[:, 9].T))

        for matri in X_test:
            t0.append(np.append(matri[0].T,matri[:,0].T))
            t1.append(np.append(matri[1].T, matri[:, 1].T))
            t2.append(np.append(matri[2].T, matri[:, 2].T))
            t3.append(np.append(matri[3].T, matri[:, 3].T))
            t4.append(np.append(matri[4].T, matri[:, 4].T))
            t5.append(np.append(matri[5].T, matri[:, 5].T))
            t6.append(np.append(matri[6].T, matri[:, 6].T))
            t7.append(np.append(matri[7].T, matri[:, 7].T))
            t8.append(np.append(matri[8].T, matri[:, 8].T))
            t9.append(np.append(matri[9].T, matri[:, 9].T))
        


        r0 = np.array(r0).reshape((8000,1,20))
        r1 = np.array(r1).reshape((8000,1,20))
        r2 = np.array(r2).reshape((8000,1,20))
        r3 = np.array(r3).reshape((8000,1,20))
        r4 = np.array(r4).reshape((8000,1,20))
        r5 = np.array(r5).reshape((8000,1,20))
        r6 = np.array(r6).reshape((8000,1,20))
        r7 = np.array(r7).reshape((8000,1,20))
        r8 = np.array(r8).reshape((8000,1,20))
        r9 = np.array(r9).reshape((8000,1,20))
        
        t0 = np.array(t0).reshape((2000, 1, 20))
        t1 = np.array(t1).reshape((2000, 1, 20))
        t2 = np.array(t2).reshape((2000, 1, 20))
        t3 = np.array(t3).reshape((2000, 1, 20))
        t4 = np.array(t4).reshape((2000, 1, 20))
        t5 = np.array(t5).reshape((2000, 1, 20))
        t6 = np.array(t6).reshape((2000, 1, 20))
        t7 = np.array(t7).reshape((2000, 1, 20))
        t8 = np.array(t8).reshape((2000, 1, 20))
        t9 = np.array(t9).reshape((2000, 1, 20))


        #Generate MLP for every input
        inp0 = Input(shape=(1,20))
        x0 = Dense(100, activation='relu')(inp0)
        #x0 = Dropout(dout)(x0)
        x0 = Dense(100, activation='relu')(x0)
        inp1 = Input(shape=(1, 20))
        x1 = Dense(100, activation='relu')(inp1)
        #x1 = Dropout(dout)(x1)
        x1 = Dense(100, activation='relu')(x1)
        inp2 = Input(shape=(1, 20))
        x2 = Dense(100, activation='relu')(inp2)
        #x2 = Dropout(dout)(x2)
        x2 = Dense(100, activation='relu')(x2)
        inp3 = Input(shape=(1, 20))
        x3 = Dense(100, activation='relu')(inp3)
        #x3 = Dropout(dout)(x3)
        x3 = Dense(100, activation='relu')(x3)
        inp4 = Input(shape=(1, 20))
        x4 = Dense(100, activation='relu')(inp4)
        #x4 = Dropout(dout)(x4)
        x4 = Dense(100, activation='relu')(x4)
        inp5 = Input(shape=(1, 20))
        x5 = Dense(100, activation='relu')(inp5)
        #x5 = Dropout(dout)(x5)
        x5 = Dense(100, activation='relu')(x5)
        inp6 = Input(shape=(1, 20))
        x6 = Dense(100, activation='relu')(inp6)
        #x6 = Dropout(dout)(x6)
        x6 = Dense(100, activation='relu')(x6)
        inp7 = Input(shape=(1, 20))
        x7 = Dense(100, activation='relu')(inp7)
        #x7 = Dropout(dout)(x7)
        x7 = Dense(100, activation='relu')(x7)
        inp8 = Input(shape=(1, 20))
        x8 = Dense(100, activation='relu')(inp8)
        #x8 = Dropout(dout)(x8)
        x8 = Dense(100, activation='relu')(x8)
        inp9 = Input(shape=(1, 20))
        x9 = Dense(100, activation='relu')(inp9)
        #x9 = Dropout(dout)(x9)
        x9 = Dense(100, activation='relu')(x9)
        concat = Concatenate()([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9])  #concatenate the variables
        flat = Flatten()(concat)
        out = Dense(1024,activation='relu')(flat)
        out = Dense(512, activation='relu')(out)
        out = Dense(256, activation='relu')(out)
        out = Dropout(0.2)(out)
        out = Dense(9,activation='sigmoid')(out)
        model = Model(inputs=[inp0,inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp8,inp9],outputs=out)
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.summary()
        model.fit([r0,r1,r2,r3,r4,r5,r6,r7,r8,r9], y_train, epochs=10, batch_size=10, verbose=1,  #Train the network
                  validation_data=([t0,t1,t2,t3,t4,t5,t6,t7,t8,t9],y_test))
        print(model.predict_on_batch([t0[:10],t1[:10],t2[:10],t3[:10],t4[:10],t5[:10],t6[:10],t7[:10],t8[:10],t9[:10]]))

        print(y_test[:10]) # Check the output


if __name__=="__main__":
    dnn = NNTSP()
    dnn.get_pickles()
    dnn.cnn_path_train()