# Machine Learning Traveling Salesman Problem and Knapsack Approximation

This research tackled the combinatorial optimization of the Traveling Salesman Problem (TSP) using various types of artificial neural networks. Multiple inputs were used, including adjacency matrices, eigenvectors. The problem was approached using Multi-Layered Perceptrons (MLP), Convolutional Neural Networks (CNN) and Pointer Networks. Different configurations of neural networks were implemented, such as Parallel input CNNs and Parallel input MLPs and a TSP approximation using pointer networks in a branch and bound algorithm.


### Prerequisites

This project contains both python .py and Jupyter notebook .ipynb files. This project uses Python 3.6 with the following libraries:

```
Tensorflow=1.15.0
```
```
 Keras=2.2.5
```
```
 numpy=1.17.4
```

### Running

The .py files may be used in Google Colaboratory if you don't have the required libraries.

This can be done by copy and pasting the code in a Colaboratory Cell. 

The Jupyter Notebooks can be uploaded and ran. 
You can run this [example](https://colab.research.google.com/drive/1bY9HB5v2sRuoX9jHnWo-Wni7qVZUINoJ) of a branch and bound using pointer networks. 


## References
Code: 
* [Aurelienbibaut](https://github.com/aurelienbibaut/Actor_CriticPointer_Network-TSP.git) - Pointer Networks code.

Please read:
 * [Pointer Networks](https://arxiv.org/abs/1506.03134) paper for details on Pointer Networks.
 * [Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis](https://arxiv.org/abs/1802.09941) for ideas in paralellism.

## Authors

**Daniel Lopez**  [webpage](https://www.daniellopez.me/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
