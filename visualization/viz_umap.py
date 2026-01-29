import matplotlib.pyplot as plt
import numpy as np

def plot_umap_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str = "umap_clusters.png",
    title: str = "UMAP Embeddings"
):
    """
    Plot 2D UMAP embeddings colored by cluster labels.

    Args:
        embeddings (np.ndarray): 2D array of shape (N, 2) with UMAP-reduced embeddings.
        labels (np.ndarray): Array of shape (N,) with cluster labels.
        save_path (str): Path to save the PNG plot.
        title (str): Title of the plot.
    """
    if embeddings.shape[1] != 2:
        raise ValueError(f"Embeddings must be 2D (N x 2), got shape {embeddings.shape}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap="tab20",
        s=20,
        alpha=0.8,
        edgecolors="k"
    )
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"UMAP plot saved to {save_path}")
