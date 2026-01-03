import numpy as np
from numpy.linalg import svd


def picard_data(A: np.ndarray, y: np.ndarray):
    U, s, _ = svd(A, full_matrices=False)
    uy = np.abs(U.T @ y)
    return s, uy
