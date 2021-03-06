from sklearn import neighbors, cluster
from sklearn.metrics import pairwise
from sklearn.manifold import SpectralEmbedding

from scipy.spatial.distance import pdist, squareform
import pygsp
import torch
import numpy as np


def create_graph_from_embedding(embedding, name, k=10, n_clusters=8):
    latent_dim, batch_size = embedding.shape
    if name =='gaussian':
        # Compute a gaussian kernel over the node activations
        node_distances = squareform(pdist(embedding, 'sqeuclidean'))
        s = 1
        K = np.exp(-node_distances / s**2)
        K[K < 0.1] = 0
        A = K * (np.ones((latent_dim, latent_dim)) - np.identity(latent_dim))
        return A
    elif name == 'knn':
        mat = neighbors.kneighbors_graph(embedding, n_neighbors=k, metric='cosine', mode='distance')
        mat.data = 1 - mat.data
        A = mat.toarray()
        A = (A + A.T) / 2
        return A
    elif name == 'knn-flat':
        A = neighbors.kneighbors_graph(embedding, n_neighbors=k, metric='cosine').toarray()
        A = (A + A.T) / 2
        return A
    elif name == 'adaptive': # It's super slow
        # Find distance of k-th nearest neighbor and set as bandwidth
        neigh = neighbors.NearestNeighbors(n_neighbors=3)
        neigh.fit(embedding)
        dist, _ = neigh.kneighbors(embedding, return_distance=True)
        kdist = dist[:,-1]
        # Apply gaussian kernel with adaptive bandwidth
        node_distances = squareform(pdist(embedding, 'sqeuclidean'))
        K = np.exp(-node_distances / kdist**2)
        A = K * (np.ones((latent_dim, latent_dim)) - np.identity(latent_dim))
        A = (A + np.transpose(A)) / 2 # Symmetrize knn graph
        return A
    elif name == 'full':
        A = pairwise.cosine_similarity(embedding)
        np.fill_diagonal(A, 0)
        return np.maximum(A, 0)
    elif name == 'hclust':
        d = pairwise.cosine_distances(embedding)
        clusts = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average").fit(d).labels_
        A = np.zeros(d.shape)
        for i in range(clusts.max() + 1):
            A[np.ix_(clusts == i, clusts == i)] = 1.0

        np.fill_diagonal(A, 0)
        return A
    elif 'knn-spectral':
        mat = neighbors.kneighbors_graph(embedding, n_neighbors=k, metric='cosine', mode='distance')
        mat.data = 1 - mat.data
        A = mat.toarray()
        A = (A + A.T) / 2
        clusts = cluster.SpectralClustering(n_clusters=n_clusters, affinity='precomputed').fit(A).labels_

        mask = np.zeros(A.shape)
        for i in range(clusts.max() + 1):
            mask[np.ix_(clusts == i, clusts == i)] = 1.0

        return (A > 1e-5) * mask
    else:
        raise RuntimeError('Unknown graph name %s' % name)

        
def create_lap_from_embedding(embedding, *args, **kwargs):
    adj_mat = create_graph_from_embedding(embedding, *args, **kwargs)
    graph = pygsp.graphs.Graph(adj_mat)
    graph.compute_laplacian(lap_type='normalized')
    return torch.Tensor(graph.L.A)


def graph_loss(activations, lap):
    return (activations.mm(lap) * activations).sum() / activations.shape[1]


def create_graph_from_layered_embedding(embs, frac:float = 0.1, n_clusters:int = 0):
    n_hidden = len(embs)
    layers = [e.shape[0] for e in embs]
    
    neighs = [neighbors.NearestNeighbors(n_neighbors=int(frac * layers[i] + 1), metric='cosine').fit(embs[i]) for i in range(n_hidden)]
    ids_per_layer = [sum(layers[:i]) + np.arange(layers[i], dtype=int) for i in range(n_hidden)]
    
    adj_mat = np.zeros((sum(layers), sum(layers)))

    for i in range(n_hidden):
        dist, ids = [x[:, 1:] for x in neighs[i].kneighbors(embs[i], return_distance=True)]
        for v1,v2s in enumerate(ids):
            adj_mat[ids_per_layer[i][v1], ids_per_layer[i][v2s]] = 1 - dist[v1,:]

        if i != n_hidden - 1:
            dist, ids = [x[:, :-1] for x in neighs[i + 1].kneighbors(embs[i], return_distance=True)]
            for v1,v2s in enumerate(ids):
                adj_mat[ids_per_layer[i][v1], ids_per_layer[i + 1][v2s]] = 1 - dist[v1,:]

        if i != 0:
            dist, ids = [x[:, :-1] for x in neighs[i - 1].kneighbors(embs[i], return_distance=True)]
            for v1,v2s in enumerate(ids):
                adj_mat[ids_per_layer[i][v1], ids_per_layer[i - 1][v2s]] = 1 - dist[v1,:]

    adj_mat = (adj_mat + adj_mat.T) / 2
    
    if n_clusters <= 0:
        return adj_mat

    clusts = cluster.SpectralClustering(n_clusters=n_clusters, affinity='precomputed').fit(adj_mat).labels_

    mask = np.zeros_like(adj_mat)
    for i in range(clusts.max() + 1):
        mask[np.ix_(clusts == i, clusts == i)] = 1.0

    return (adj_mat > 1e-5) * mask
