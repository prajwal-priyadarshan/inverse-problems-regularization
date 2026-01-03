import numpy as np
from ..reconstruction import pseudoinverse, tikhonov, tsvd
from .error_metrics import mse, psnr, relative_error


def compare_methods(A: np.ndarray, y: np.ndarray, x_true: np.ndarray, tikh_lambdas, tsvd_ks):
    results = {
        "pseudoinverse": {},
        "tikhonov": [],
        "tsvd": [],
    }
    x_pinv = pseudoinverse.reconstruct(A, y)
    results["pseudoinverse"] = {
        "mse": mse(x_true, x_pinv),
        "psnr": psnr(x_true, x_pinv),
        "rel_error": relative_error(x_true, x_pinv),
    }
    for lam in tikh_lambdas:
        x_hat = tikhonov.reconstruct(A, y, lam)
        results["tikhonov"].append({
            "lambda": float(lam),
            "mse": mse(x_true, x_hat),
            "psnr": psnr(x_true, x_hat),
            "rel_error": relative_error(x_true, x_hat),
        })
    for k in tsvd_ks:
        x_hat = tsvd.reconstruct(A, y, k)
        results["tsvd"].append({
            "k": int(k),
            "mse": mse(x_true, x_hat),
            "psnr": psnr(x_true, x_hat),
            "rel_error": relative_error(x_true, x_hat),
        })
    return results
