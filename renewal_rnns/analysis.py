from __future__ import annotations
import numpy as np


def pca_reduce(X: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Simple PCA via SVD (no sklearn dependency).

    Args:
        X: (n_samples, n_features) array
        k: number of principal components

    Returns:
        Z: (n_samples, k) principal component scores
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :k] * S[:k]
    return Z
