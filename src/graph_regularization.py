from sklearn import neighbors
from sklearn.metrics import pairwise
from scipy.spatial.distance import pdist, squareform
import pygsp
import torch


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
        A = (A + A.T) / 2 # Symmetrize knn graph
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
    else:
        raise RuntimeError('Unknown graph name %s' % name)

        
def create_lap_from_embedding(embedding, *args, **kwargs):
    adj_mat = create_graph_from_embedding(embedding, *args, **kwargs)
    graph = pygsp.graphs.Graph(adj_mat)
    graph.compute_laplacian(lap_type='normalized')
    return torch.Tensor(graph.L.A)


def graph_loss(activations, lap):
    return (activations.mm(lap) * activations).sum() / activations.shape[1]