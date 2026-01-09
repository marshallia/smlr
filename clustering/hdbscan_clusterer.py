import numpy as np
import hdbscan
from typing import Tuple, Optional


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: Optional[int] = None,
    cluster_selection_method: str = "eom",
    return_probabilities: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run HDBSCAN clustering on reduced embeddings.

    Args:
        embeddings (np.ndarray): Input array of shape (N, D) with reduced embeddings.
        min_cluster_size (int): Minimum size of clusters.
        min_samples (int | None): Minimum samples for core points. If None, defaults to min_cluster_size.
        cluster_selection_method (str): "eom" or "leaf".
        return_probabilities (bool): If True, also return cluster membership probabilities.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]:
            - labels: Cluster labels of shape (N,), noise labeled as -1.
            - probabilities: Optional array of shape (N,) with cluster membership probabilities.
    """
    # Input validation
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"embeddings must be a numpy.ndarray, got {type(embeddings)}")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D (N x D), got shape {embeddings.shape}")
    if cluster_selection_method not in ("eom", "leaf"):
        raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")

    # Initialize HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method
    )

    # Fit and predict
    labels = clusterer.fit_predict(embeddings)

    if return_probabilities:
        probabilities = clusterer.probabilities_
        return labels, probabilities
    else:
        return labels, None
