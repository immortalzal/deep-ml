import numpy as np


def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    X_ = np.array(X)
    A = np.dot(X_, X_.T)
    exp = np.exp(A - np.max(A, axis = -1, keepdims = True))
    weight = exp / np.sum(exp, axis = -1, keepdims = True)
    return np.dot(weight, X)
    """
    Compute context vectors using simple self-attention (no trainable weights).

    Args:
        X: Input embeddings of shape (T, d) as a list of lists.

    Returns:
        Context vectors of shape (T, d) as a list of lists.
    """
    pass
