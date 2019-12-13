# Machine Learning Traveling Salesman Problem and Knapsack Approximation


This [research project](https://drive.google.com/file/d/1zufq8r2DzvAyCCHte3wBOf_HJJQN0nef/view?usp=sharing) tackled the combinatorial optimization of the Traveling Salesman Problem (TSP) using various types of artificial neural networks. We developed a branch and bound algorithm which takes in a neural network that can generate a heuristic of the path or the cost of the TSP. This required multiple inputs such as adjacency matrices, eigenvectors, and combinations of the two. The problem was approached using Multi-Layered Perceptrons (MLP), Convolutional Neural Networks (CNN), Long short-term memory networks(LSTM) and Pointer Networks. Different configurations of neural networks were implemented, such as Parallel input CNNs and Parallel input MLPs to generate a TSP approximation used in a branch and bound algorithm.

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
#### direct_mlp_cnn_lstm_eigen.py
This file is used to test multiple neural networks with similar hyperparameters
Call the network you want to train and save the model. Each network will show the results as a print.
You may change the hyperparameters in the file, and change eigen to false if you want to use Adjacency matrices as input.
```
dnn = NNTSP()
dnn.get_pickles(eigen = True) 
dnn.mlp_tsp_train()
dnn.cnn_tsp_train()
dnn.lstm_tsp_train()
```

##### mlp_tsp_dif_loss.py
This file was used to test different loss functions using adjacency matrices. To use type and run:  
```
dnn = NNTSP()
dnn.get_pickles()
dnn.mlp_loss_tsp_train()
```

#### Pointer-Networks
I have set the pointer network and dynamic programming in the dynamic_path_pointer-net.ipynb which can be opened in Google Colab for testing.
##### Google Colab Jupyter Notebooks
See "To use Google Colab Juypter Notebooks.pdf" for reference.

You can run the latest file [here](https://colab.research.google.com/drive/1bY9HB5v2sRuoX9jHnWo-Wni7qVZUINoJ) of branch and bound using pointer networks. 


## References
Code: 
* [Aurelienbibaut](https://github.com/aurelienbibaut/Actor_CriticPointer_Network-TSP.git) - Pointer Networks code.

Please read:
 * [Pointer Networks](https://arxiv.org/abs/1506.03134) paper for details on Pointer Networks.
 * [Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis](https://arxiv.org/abs/1802.09941) for ideas in parallelism.

## Authors

* Daniel Lopez
* Sanjiv Kapoor
