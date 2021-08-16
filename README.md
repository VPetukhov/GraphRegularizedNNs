# Graph-regularized Neural Networks

The repo contains application of my extention of the method from [Interpretable Neuron Structuring with Graph Spectral Regularization by Tong, et. al](https://arxiv.org/pdf/1810.00424.pdf)
to a few other types of networks (more realistic in my opinion).

The main results are:

- [w2v_regularized.ipynb](./w2v_regularized.ipynb)
  - Enforcing grid structure on the hidden layer of word2vec allows visual interpretation of the word embedding without loss in quality ("Graph-regularized (grid)" section)
  - To get actual clustering structure on the word2vec I had to enforce fixed number of modules into the learned graph [using spectral clustering](https://github.com/VPetukhov/GraphRegularizedNNs/blob/main/src/graph_regularization.py#L56). 
  It allowed to get a nice clustered structure of the hidden layer ("Grap-regularized: 6 clusters spectral" part), though my quick attempt to interpret it weren't succesful.
- [mlp_coactivation_reg_v2.ipynb](./mlp_coactivation_reg_v2.ipynb): trying to add modular structure across several layers for an MLP on FashionMNIST reduced **both** its performance and its interpretability.
