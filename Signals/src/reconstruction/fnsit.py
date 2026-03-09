"""
Fast Non-Stationary Iterated Tikhonov (FNSIT) Regularization

FNSIT is an efficient iterative regularization method that:
1. Uses non-stationary regularization (decreasing α_n) like NSIT
2. Replaces exact linear solve with fast approximate solver (gradient descent)
3. Maintains automatic stopping via Morozov discrepancy principle

Key advantage: Computational speedup (O(mn*k_inner)) vs NSIT's O(n^3) exact solve

Mathematical Foundation:

Standard NSIT update rule:
    x_n = x_{n-1} + (A^T A + α_n I)^{-1} A^T r_{n-1}

where:
    r_{n-1} = y - A x_{n-1}  (residual)
    α_n = α_0 * q^n          (decreasing regularization)

FNSIT innovation: Replace exact solve with gradient descent approximation:
    Minimize: f(z) = (1/2)||Az||² + (α/2)||z||² - ⟨rhs, z⟩
    Gradient: ∇f(z) = (A^T A + α I)z - rhs
    Update: z ← z - β ∇f(z)  (iterate k_inner times)

Morozov Stopping: ||y - Ax_n|| ≈ τ*δ*||y||
"""

import numpy as np
from numpy.linalg import norm


def fnsit_fast_solver(ATA: np.ndarray, rhs: np.ndarray, alpha: float,
                      inner_steps: int = 5, step_size: float = 0.1) -> np.ndarray:
    """
    Fast approximate solver for: (A^T A + α I) z = rhs

    Uses gradient descent instead of exact linear solve:
        Minimize: f(z) = (1/2)||Az||² + (α/2)||z||² - ⟨rhs, z⟩
        Gradient: ∇f(z) = (A^T A + α I)z - rhs
        Update: z ← z - β * ∇f(z)

    Parameters:
    -----------
    ATA : np.ndarray
        Precomputed A^T A matrix (n × n)
    rhs : np.ndarray
        Right-hand side A^T r vector (length n)
    alpha : float
        Regularization parameter
    inner_steps : int, default=5
        Number of gradient descent iterations (few are enough!)
    step_size : float, default=0.1
        Step size β for gradient descent

    Returns:
    --------
    z : np.ndarray
        Approximate solution to (A^T A + α I) z = rhs
    """
    n = len(rhs)
    z = np.zeros(n)  # Cold start

    for _ in range(inner_steps):
        # Gradient = (A^T A + α I)z - rhs
        gradient = (ATA + alpha * np.eye(n)) @ z - rhs

        # Gradient descent step
        z = z - step_size * gradient

    return z


def fnsit_reconstruct(A: np.ndarray, y: np.ndarray,
                      alpha0: float, q: float, max_iter: int,
                      tau: float, delta: float,
                      inner_steps: int = 5, step_size: float = 0.1,
                      x_true: np.ndarray | None = None) -> tuple:
    """
    Fast NSIT Reconstruction with Morozov Stopping

    Iteratively refines solution using approximate regularized corrections
    with decreasing regularization parameter. Stops automatically when residual
    reaches approximately τ*δ*||y||.

    Parameters:
    -----------
    A : np.ndarray
        Forward operator matrix (m × n)
    y : np.ndarray
        Measurement vector (length m)
    alpha0 : float
        Initial regularization parameter
    q : float
        Decay rate: α_n = α_0 * q^n (typically 0.8-0.95)
    max_iter : int
        Maximum number of iterations
    tau : float
        Safety factor for Morozov principle (typically 1.0-1.1)
    delta : float
        Noise level (relative noise or absolute standard deviation)
    inner_steps : int, default=5
        Number of gradient descent steps per iteration
    step_size : float, default=0.1
        Step size for gradient descent
    x_true : np.ndarray | None, optional
        True solution (for error tracking only)

    Returns:
    --------
    x : np.ndarray
        Reconstructed solution
    history : dict
        Contains:
        - 'residuals': ||y - Ax_n|| at each iteration
        - 'errors': ||x_n - x_true|| / ||x_true|| (if x_true provided)
        - 'alphas': regularization parameters used
        - 'iterations': iteration indices
        - 'final_iteration': stopping iteration
    """
    m, n = A.shape
    x = np.zeros(n)

    # Precompute A^T A (reuse across iterations)
    ATA = A.T @ A

    # Compute Morozov stopping threshold
    y_norm = norm(y)
    threshold = tau * delta * y_norm

    # Initialize history
    history = {
        'residuals': [],
        'errors': [],
        'alphas': [],
        'iterations': []
    }

    # Main iteration loop
    for iteration in range(max_iter):
        # Current regularization parameter (geometric decay)
        alpha_n = alpha0 * (q ** iteration)

        # Compute residual
        residual = y - A @ x
        residual_norm = norm(residual)

        # **Morozov Stopping Criterion**
        if residual_norm <= threshold:
            print(f"  FNSIT Morozov stopping at iteration {iteration}")
            break

        # **Fast approximate solve** (key difference from NSIT)
        rhs = A.T @ residual
        z = fnsit_fast_solver(ATA, rhs, alpha_n, inner_steps, step_size)

        # Update solution
        x = x + z

        # Track progress
        history['residuals'].append(residual_norm)
        history['alphas'].append(alpha_n)
        history['iterations'].append(iteration)

        # Optional: compute relative error if true solution provided
        if x_true is not None:
            error = norm(x - x_true) / norm(x_true)
            history['errors'].append(error)

    # Record final iteration
    history['final_iteration'] = iteration

    return x, history
