import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import tsp

class GraphGenerator:
    def __init__(self,size,max_val): # Useful public variables
        self.dimension = size
        self.max_val = max_val
        self.out_val = 0

    #Generates an undirected graph
    def undirected_graph(self):
        self.matrix = np.triu(np.random.randint(-self.max_val / 2, self.max_val, size=(self.dimension, self.dimension)).clip(min=0))
        self.matrix += self.matrix.T
        np.fill_diagonal(self.matrix, self.out_val)
        self.matrix[self.matrix < 1] = self.out_val
    #Generates a directed graph
    def directed_graph(self):
        self.matrix = np.random.randint(-self.max_val / 2, self.max_val, size=(self.dimension, self.dimension)).clip(min=0)
        np.fill_diagonal(self.matrix, self.out_val)
        self.matrix[self.matrix < 1] = self.out_val

    # Given an adjacency matrix and a path, get the cost of the path
    def get_tsp_vals(self, path, g):
        last_place = 0
        path_vals = []
        path = list(path) + [0]
        for value in path:
            path_vals.append(g[last_place][value])
            last_place = value
        return path_vals

    # get transposed adjacency matrix
    def tdirected_graph(self):
        return self.matrix.T

    # In case you want to create your own adjacency matrix
    def set_graph(self,matrix):
        self.matrix = matrix

    # Return adjacency matrix
    def get_adjacency_matrix(self):
        return self.matrix

    # Get node connections from the adjacency matrix
    def get_connections(self):
        edges = []
        for offset,row in enumerate(self.matrix):
            for i in range(len(row)-offset):
                real_index = i+offset
                weight = row[real_index]
                if(weight!=0):
                    edges.append((offset,real_index,weight))

        return edges

    # Plot the garph using networkx library
    def plot_graph(self):
        edges= self.get_connections()

        G = nx.Graph()
        for edge in edges:
            G.add_edge(edge[0],edge[1],weight = edge[2])
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G,pos,node_size=300)
        nx.draw_networkx_edges(G,pos,width=1)
        nx.draw_networkx_labels(G, pos, font_size=14, font_family='sans-serif')
        nx.draw_networkx_edge_labels(G,pos,font_size=5)
        plt.axis('off')
        plt.show()

    # Get eigen values from an adjacency matrix
    def get_eigen_values(self):
        eigens = np.linalg.eig(self.matrix)
        return eigens[0],eigens[1]

