import torch

def get_device():
    """
    Returns the appropriate torch.device (CUDA if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    """
    Sets random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)