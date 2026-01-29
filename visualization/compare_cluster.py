import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_clusters_vs_ground_truth(
    embeddings_2d: np.ndarray,
    predicted_labels: np.ndarray,
    ground_truth_labels: np.ndarray,
    output_path: str
) -> None:
    """
    Visualize UMAP embeddings colored by predicted cluster labels and ground truth labels.

    Creates two side-by-side scatter plots:
      1) Predicted Clusters
      2) Ground Truth Labels

    Args:
        embeddings_2d (np.ndarray): 2D embeddings of shape (N, 2).
        predicted_labels (np.ndarray): Predicted cluster labels of shape (N,).
        ground_truth_labels (np.ndarray): Ground truth labels of shape (N,).
        output_path (str): Path to save the PNG figure.

    Notes:
        - Noise points (label = -1) are shown in gray.
        - Legends are included if the number of unique labels <= 15.
    """
    # Input validation
    if not isinstance(embeddings_2d, np.ndarray):
        raise TypeError("embeddings_2d must be a numpy.ndarray")
    if embeddings_2d.ndim != 2 or embeddings_2d.shape[1] != 2:
        raise ValueError(f"embeddings_2d must be of shape (N, 2), got {embeddings_2d.shape}")

    if not isinstance(predicted_labels, np.ndarray):
        raise TypeError("predicted_labels must be a numpy.ndarray")
    if not isinstance(ground_truth_labels, np.ndarray):
        raise TypeError("ground_truth_labels must be a numpy.ndarray")

    if embeddings_2d.shape[0] != predicted_labels.shape[0] or embeddings_2d.shape[0] != ground_truth_labels.shape[0]:
        raise ValueError("embeddings_2d, predicted_labels, and ground_truth_labels must have the same length")

    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def _plot(ax, labels, title):
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            mask = labels == lbl
            if lbl == -1:
                # Noise points
                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c="lightgray",
                    s=20,
                    label="Noise (-1)",
                    alpha=0.6,
                    edgecolors="none"
                )
            else:
                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    s=20,
                    label=str(lbl),
                    alpha=0.8,
                    edgecolors="none"
                )
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        if len(unique_labels) <= 15:
            ax.legend(markerscale=2, fontsize=8)

    # Plot predicted clusters
    _plot(axes[0], predicted_labels, "Predicted Clusters")

    # Plot ground truth labels
    _plot(axes[1], ground_truth_labels, "Ground Truth Labels")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Cluster comparison plot saved to {output_path}")
