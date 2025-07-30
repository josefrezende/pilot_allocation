"""Simplified max-min spectral efficiency routine for cell-free massive MIMO.

This module contains a tiny demonstration of a WMMSE based bisection search for
the max–min signal to interference and noise ratio (SINR).  The implementation
is intentionally lightweight so it can be run without any dependencies other
than :mod:`numpy` and the Python standard library.
"""

import argparse
import numpy as np  # numerical operations

def rate_from_sinr(sinr):
    """Return spectral efficiency values from SINR."""

    # capacity of a single‑user Gaussian channel in bit/s/Hz
    return np.log2(1 + sinr)

def compute_sinr(mu, A, B, sigma2):
    """Compute the SINR for each user.

    Parameters
    ----------
    mu : ndarray, shape (K, L)
        Current power allocation matrix.
    A : list of ndarray
        Channel vectors ``a_k`` for each user ``k``.
    B : list of list of ndarray
        Matrices ``B_ki`` describing interference coupling.
    sigma2 : float
        Noise power.

    Returns
    -------
    ndarray
        The SINR value for each user.
    """
    K = len(A)
    SINRs = np.zeros(K)

    # evaluate each user separately
    for k in range(K):
        # numerator: desired signal power
        num = (A[k] @ mu[k]) ** 2

        # denominator: interference plus noise
        denom = 0
        for i in range(K):
            denom += mu[i].T @ B[k][i] @ mu[i]
        denom = denom - num + sigma2

        # resulting SINR for user k
        SINRs[k] = num / denom
    return SINRs

def wmmse_feasibility(
    mu_init,
    A,
    B,
    sigma2,
    Pmax,
    gamma_target,
    max_iter=50,
    tol=1e-3,
    step=0.01,
):
    """Check whether a target SINR is achievable using a simple WMMSE loop."""
    K, L = mu_init.shape
    # working copy of the power allocation matrix
    mu = mu_init.copy()

    for _ in range(max_iter):
        # keep previous iterate to check convergence
        mu_prev = mu.copy()

        # Step 1: Compute v_k and e_k for each user
        v = np.zeros(K)
        e = np.zeros(K)
        for k in range(K):
            # linear receiver for user k
            num = A[k] @ mu[k]
            denom = sigma2
            for i in range(K):
                denom += mu[i].T @ B[k][i] @ mu[i]
            v[k] = num / denom

            # estimation error
            e[k] = 1 - (num ** 2) / denom

        # Step 2: Fix omega_k = 1 / e_k for all k (WMMSE weights)
        omega = 1 / e

        # Step 3: Simplified update for mu via projected gradient
        for l in range(L):
            # gradient for the l‑th antenna powers
            grad = np.zeros(K)
            for k in range(K):
                grad[k] = -2 * omega[k] * v[k] * A[k][l]

            # projected gradient step under per‑antenna power constraint
            mu[:, l] = mu[:, l] + step * grad
            norm_sq = np.sum(mu[:, l] ** 2)
            if norm_sq > Pmax:
                mu[:, l] *= np.sqrt(Pmax / norm_sq)

        # Step 4: Check convergence
        delta = np.linalg.norm(mu - mu_prev)
        if delta < tol:
            break

    # Compute SINRs with final mu
    SINRs = compute_sinr(mu, A, B, sigma2)
    return np.all(SINRs >= gamma_target), mu

def max_min_rate_wmmse(
    A,
    B,
    sigma2,
    Pmax,
    mu_init,
    gamma_min=1e-3,
    gamma_max=20,
    tol=1e-3,
):
    """Return the best minimum SINR using a feasibility‑check bisection loop."""
    K = len(A)
    mu_opt = None

    while gamma_max - gamma_min > tol:
        gamma_mid = (gamma_min + gamma_max) / 2
        feasible, mu_candidate = wmmse_feasibility(
            mu_init, A, B, sigma2, Pmax, gamma_mid
        )
        if feasible:
            gamma_min = gamma_mid
            mu_opt = mu_candidate
        else:
            gamma_max = gamma_mid

    return gamma_min, mu_opt


def main():
    """Run a small randomised example for demonstration."""
    parser = argparse.ArgumentParser(description="Max-min SINR via WMMSE")
    parser.add_argument("K", type=int, nargs="?", default=4, help="number of users")
    parser.add_argument("L", type=int, nargs="?", default=3, help="number of APs")
    parser.add_argument("sigma2", type=float, nargs="?", default=1e-3, help="noise power")
    parser.add_argument("Pmax", type=float, nargs="?", default=1.0, help="power constraint")
    args = parser.parse_args()

    rng = np.random.default_rng()
    A = [rng.random(args.L) for _ in range(args.K)]
    B = [[rng.random((args.L, args.L)) for _ in range(args.K)] for _ in range(args.K)]
    mu_init = rng.random((args.K, args.L))

    gamma_star, mu_opt = max_min_rate_wmmse(A, B, args.sigma2, args.Pmax, mu_init)

    rates = rate_from_sinr(compute_sinr(mu_opt, A, B, args.sigma2))
    print(f"Max-min SINR achieved: {gamma_star:.4f}")
    print("Rates:", np.round(rates, 4))


if __name__ == "__main__":  # pragma: no cover - simple CLI
    main()
