import numpy as np
from pathlib import Path

def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)
