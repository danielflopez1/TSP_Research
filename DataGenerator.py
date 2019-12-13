import numpy as np
import tsp
import GraphGenerator
import pickle
import time


class Data_Generator:
    def __init__(self):  # set initial variables for data generation
        self.change = 0
        self.size = 5 
        self.max_value = 100  

    def tsp_data(self):
        xs = []
        ys = []
        for x in range(self.change,1000000):  #Generate 1M datapoints
            GG = GraphGenerator.GraphGenerator(self.size, self.max_value)
            GG.undirected_graph()
            g = np.array(GG.get_adjacency_matrix())  #Create undirected adjacency matrix
            matrx = g.copy()
            g[g==0] = 999
            self.path = tsp.tsp_dp_solve(g)  #solve the tsp
            self.tsp_val = tsp.tour_len(self.path, g) #get the tsp cost
            ys.append(self.tsp_val)             #Add the data to the arrays
            xs.append(list(matrx.flatten()))
            self.change +=1
            
        pickle_file=open('m5tspxs.pickle','wb') #Pickle the data for later use
        pickle.dump(xs,pickle_file)
        pickle_file.close()

        pickle_file= open('m5tspys.pickle', 'wb')
        pickle.dump(ys,pickle_file)
        pickle_file.close()

    def get_pickles(self):                      #check the Pickle data
        pickle_file = open('xs.pickle', 'rb')
        datas = pickle.load(pickle_file)
        pickle_file.close()
        print(datas)

        pickle_file = open('ys.pickle', 'rb')
        ydatas = pickle.load(pickle_file)
        pickle_file.close()
        print("---",ydatas)
gm = Data_Generator()
gm.tsp_data()
gm.get_pickles()
