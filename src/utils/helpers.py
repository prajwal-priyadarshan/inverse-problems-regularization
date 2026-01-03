import os
from pathlib import Path

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def data_dir() -> Path:
    return project_root() / "data" / "synthetic"

def results_dir() -> Path:
    return project_root() / "results"
