import numpy as np
from sklearn.cluster import KMeans

class KMeansClusterer:
    def __init__(self, n_clusters: int, random_state: int = 42):
        """
        KMeans clustering module.

        Args:
            n_clusters (int): Number of clusters.
            random_state (int): Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            random_state=random_state,
            n_init="auto"
        )

    def fit_predict(self, embeddings: np.ndarray, return_centroids: bool = False):
        """
        Fit KMeans on embeddings and return cluster labels.

        Args:
            embeddings (np.ndarray): Reduced embeddings of shape (N, D).
            return_centroids (bool): If True, also return cluster centroids.

        Returns:
            labels (np.ndarray): Cluster labels of shape (N,).
            centroids (np.ndarray, optional): Cluster centroids of shape (n_clusters, D).
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        labels = self.model.fit_predict(embeddings)

        if return_centroids:
            return labels, self.model.cluster_centers_
        return labels
