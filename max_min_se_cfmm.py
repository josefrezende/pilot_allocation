import numpy as np

def compute_sinr(mu, A, B, sigma2):
    """
    Compute SINR_k = (a_k^T mu_k)^2 / (sum_i mu_i^T B_ki mu_i - (a_k^T mu_k)^2 + sigma^2)
    """
    K = len(A)
    SINRs = np.zeros(K)
    for k in range(K):
        num = (A[k] @ mu[k]) ** 2
        denom = 0
        for i in range(K):
            denom += mu[i].T @ B[k][i] @ mu[i]
        denom = denom - num + sigma2
        SINRs[k] = num / denom
    return SINRs

def wmmse_feasibility(mu_init, A, B, sigma2, Pmax, gamma_target, max_iter=50, tol=1e-3):
    """
    WMMSE Feasibility solver: checks if a given gamma_target is achievable
    """
    K, L = mu_init.shape
    mu = mu_init.copy()

    for _ in range(max_iter):
        mu_prev = mu.copy()

        # Step 1: Compute v_k and e_k for each user
        v = np.zeros(K)
        e = np.zeros(K)
        for k in range(K):
            num = A[k] @ mu[k]
            denom = sigma2
            for i in range(K):
                denom += mu[i].T @ B[k][i] @ mu[i]
            v[k] = num / denom
            e[k] = 1 - (num ** 2) / denom

        # Step 2: Fix omega_k = 1 / e_k for all k (WMMSE weights)
        omega = 1 / e

        # Step 3: ADMM-like update for mu
        # Simplified update: gradient projection (in practice, use closed-form ADMM)
        for l in range(L):
            grad = np.zeros(K)
            for k in range(K):
                grad[k] = -2 * omega[k] * v[k] * A[k][l]
            mu[:, l] = mu[:, l] + 0.01 * grad
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

def max_min_rate_wmmse(A, B, sigma2, Pmax, mu_init, gamma_min=1e-3, gamma_max=20, tol=1e-3):
    """
    Bisection to find max-min SINR using WMMSE feasibility
    """
    K = len(A)
    mu_opt = None

    while gamma_max - gamma_min > tol:
        gamma_mid = (gamma_min + gamma_max) / 2
        feasible, mu_candidate = wmmse_feasibility(mu_init, A, B, sigma2, Pmax, gamma_mid)
        if feasible:
            gamma_min = gamma_mid
            mu_opt = mu_candidate
        else:
            gamma_max = gamma_mid

    return gamma_min, mu_opt


K, L = 4, 3  # 4 users, 3 APs
A = [np.random.rand(L) for _ in range(K)]
B = [[np.random.rand(L, L) for _ in range(K)] for _ in range(K)]
sigma2 = 1e-3
Pmax = 1.0
mu_init = np.random.rand(K, L)

gamma_star, mu_opt = max_min_rate_wmmse(A, B, sigma2, Pmax, mu_init)
print(f"Max-min SINR achieved: {gamma_star:.4f}")
