# Machine Learning Traveling Salesman Problem and Knapsack Approximation

This [research project](https://drive.google.com/file/d/1zufq8r2DzvAyCCHte3wBOf_HJJQN0nef/view?usp=sharing) tackled the combinatorial optimization of the Traveling Salesman Problem (TSP) using various types of artificial neural networks. Multiple inputs were used, including adjacency matrices, eigenvectors. The problem was approached using Multi-Layered Perceptrons (MLP), Convolutional Neural Networks (CNN) and Pointer Networks. Different configurations of neural networks were implemented, such as Parallel input CNNs and Parallel input MLPs and a TSP approximation using pointer networks in a branch and bound algorithm.

## Prerequisites
```
Tensorflow
Keras
numpy
pickle
tsp
networkx
```

## Running and Testing
### Generate Data
Use the DataGenerator.py by setting the size of the graph you want, number of data_points you need and if the adjacency matrices will be directed or undirected. 
```
gm = Data_Generator(size = number_of_nodes, num_data = 1000000, directed = False)
gm.tsp_data()
```
##### Output
This will generate a pickle file of Adjacency matrices, paths and costs.

### Neural networks .py files
Can be oppened and tested after generating the data as 
```
gm = Data_Generator(size = 10, num_data = 1000000, directed = False)
gm.tsp_data()
```
### Google Colab Jupyter Notebooks
See "To use Google Colab Juypter Notebooks.pdf" for reference

You can run the latest [example](https://colab.research.google.com/drive/1bY9HB5v2sRuoX9jHnWo-Wni7qVZUINoJ) of a branch and bound using pointer networks. 


## References
Code: 
* [Aurelienbibaut](https://github.com/aurelienbibaut/Actor_CriticPointer_Network-TSP.git) - Pointer Networks code.

Please read:
 * [Pointer Networks](https://arxiv.org/abs/1506.03134) paper for details on Pointer Networks.
 * [Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis](https://arxiv.org/abs/1802.09941) for ideas in paralellism.

## Authors

* Daniel Lopez
* Sanjiv Kapoor

## Notes
This research is in progress

