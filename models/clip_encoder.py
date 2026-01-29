import torch
import numpy as np
from typing import List
from PIL import Image
import clip
from tqdm import tqdm

# Load CLIP model once at import
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """L2 normalize along the last dimension."""
    return x / x.norm(dim=-1, keepdim=True)


def encode_images(image_paths: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Encode a list of image file paths into CLIP embeddings.
    
    Args:
        image_paths: List of paths to image files.
        batch_size: Number of images per batch.
    
    Returns:
        np.ndarray of shape (len(image_paths), embedding_dim)
    """
    embeddings = []
    MODEL.eval()

    with torch.no_grad():
      for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i:i+batch_size]
            images = [PREPROCESS(Image.open(p).convert("RGB")) for p in batch_paths]
            image_tensor = torch.stack(images).to(DEVICE)

            feats = MODEL.encode_image(image_tensor)
            feats = _l2_normalize(feats)
            embeddings.append(feats.cpu().numpy())

    return np.vstack(embeddings)


def encode_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Encode a list of text descriptions into CLIP embeddings.
    
    Args:
        texts: List of text strings.
        batch_size: Number of texts per batch.
    
    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
    """
    embeddings = []
    MODEL.eval()
    # embedded_text =[]
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
            batch_sents = texts[i:i+batch_size]
            tokens = clip.tokenize(batch_sents).to(DEVICE) 
            feats = MODEL.encode_text(tokens) 
            feats = _l2_normalize(feats) 
            embeddings.append(feats.cpu().numpy())
    return np.vstack(embeddings)
