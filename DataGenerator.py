import numpy as np
import tsp
import GraphGenerator
import pickle
import time


class Data_Generator:
    def __init__(self, size, num_data, directed = True):  # set initial variables for data generation
        self.size = size 
        self.max_value = 100  
        self.num_data = num_values

    def tsp_data(self):
        xs = []
        ys = []
        yps = []
        for x in range(self.change,self.num_data):  #Generate 1M datapoints
            GG = GraphGenerator.GraphGenerator(self.size, self.max_value)
            if directed:
                GG.directed_graph() #check if adjacency matrix is directed or undirected
            else:
                GG.undirected_graph()
            g = np.array(GG.get_adjacency_matrix())  #Create undirected adjacency matrix
            matrx = g.copy()
            g[g==0] = 999
            path = tsp.tsp_dp_solve(g)  #solve the tsp
            tsp_val = tsp.tour_len(self.path, g) #get the tsp cost
            yps.append(path)
            ys.append(tsp_val)             #Add the data to the arrays
            xs.append(list(matrx.flatten()))
            
        pickle_file=open('xs.pickle','wb') #Pickle the data for later use
        pickle.dump(xs,pickle_file)
        pickle_file.close()
        
        pickle_file= open('yps.pickle', 'wb')
        pickle.dump(ys,pickle_file)
        pickle_file.close()
        
        pickle_file= open('ys.pickle', 'wb')
        pickle.dump(ys,pickle_file)
        pickle_file.close()

    def get_pickles(self):                      #check the Pickle data
        pickle_file = open('xs.pickle', 'rb')
        datas = pickle.load(pickle_file)
        pickle_file.close()
        print(datas)

        pickle_file = open('yps.pickle', 'rb')
        ypathdatas = pickle.load(pickle_file)
        pickle_file.close()
        print(ypathdatas)
        
        pickle_file = open('ys.pickle', 'rb')
        ydatas = pickle.load(pickle_file)
        pickle_file.close()
        print("---",ydatas)

