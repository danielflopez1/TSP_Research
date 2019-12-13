import numpy as np
import tsp
import GraphGenerator
import pickle
import time


class game:
    def __init__(self):
        self.change = 0
        self.size = 5  # np.random.randint(5,10)
        self.max_value = 100

    def tsp_game(self):
        xs = []
        ys = []
        for x in range(self.change,1000000):
            if(self.change>1000000):
                break
            if(self.change%500 ==0):
                print(self.change)
            GG = GraphGenerator.GraphGenerator(self.size, self.max_value)
            GG.undirected_graph()
            g = np.array(GG.get_adjacency_matrix())
            matrx = g.copy()
            g[g==0] = 999
            #print(matrx)
            self.path = tsp.tsp_dp_solve(g)

            self.tsp_val = tsp.tour_len(self.path, g)

            ys.append(self.tsp_val)
            xs.append(list(matrx.flatten()))
            #print(xs, ys)
            #input("")
            self.change +=1
        #print(xs)
        #print(ys)
        pickle_file=open('m5tspxs.pickle','wb')
        pickle.dump(xs,pickle_file)
        pickle_file.close()

        pickle_file= open('m5tspys.pickle', 'wb')
        pickle.dump(ys,pickle_file)
        pickle_file.close()

    def get_pickles(self):
        pickle_file = open('xs.pickle', 'rb')
        datas = pickle.load(pickle_file)
        pickle_file.close()
        print(datas)

        pickle_file = open('ys.pickle', 'rb')
        ydatas = pickle.load(pickle_file)
        pickle_file.close()
        print("---",ydatas)
gm = game()
gm.tsp_game()
gm.get_pickles()
