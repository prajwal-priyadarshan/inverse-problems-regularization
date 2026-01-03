import numpy as np
from .svd_analysis import condition_number as _cond

def condition_number(A: np.ndarray) -> float:
    return _cond(A)
