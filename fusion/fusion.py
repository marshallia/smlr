import numpy as np

def fuse_embeddings(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    method: str = "concat",
    weight: float = 0.5
) -> np.ndarray:
    """
    Fuse multimodal embeddings (image + text).

    Args:
        image_embeddings (np.ndarray): Array of shape (N, D) for image embeddings.
        text_embeddings (np.ndarray): Array of shape (N, D) for text embeddings.
        method (str): Fusion method, either "concat" or "weighted".
        weight (float): Weight for text embeddings when method="weighted".
                        Image weight = (1 - weight).

    Returns:
        np.ndarray: Fused embeddings.
            - If method="concat": shape (N, 2D)
            - If method="weighted": shape (N, D)

    Raises:
        ValueError: If shapes mismatch or method is invalid.
    """
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError(
            f"Shape mismatch: image {image_embeddings.shape}, text {text_embeddings.shape}"
        )

    if method == "concat":
        fused = np.concatenate([image_embeddings, text_embeddings], axis=1)
    elif method == "weighted":
        fused = (1 - weight) * image_embeddings + weight * text_embeddings
    else:
        raise ValueError(f"Unsupported fusion method: {method}")

    return fused
