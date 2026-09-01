"""Lanne, Luoto and Saikkonen (2012) simulation-based predictive density."""

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm

from src.stable_mar.stable import stable_pdf, stable_rvs
from src.stable_mar.stable_mar import madelta


def lls2012_predictive_density_given_yt(
    psi_hat: list,
    phi_hat: list,
    alpha_hat: float,
    beta_hat: float,
    sigma_hat: float,
    y: list,
    horizon: int,
    M: int,
    N_draws: int,
    grid_y: np.ndarray,
    seed: int = None,
):
    """Lanne, Luoto and Saikkonen (2012) predictive density at horizon h."""
    s = len(psi_hat)
    r = len(phi_hat)

    # 1) simulate future shocks
    eps = stable_rvs(
        alpha=alpha_hat,
        beta=beta_hat,
        loc=0,
        scale=sigma_hat,
        size=(M, N_draws),
        seed=seed,
    )

    # 2) Compute e_T
    beta_offdiag = np.real(
        np.array(
            [madelta(cvec=[0.0], ncvec=psi_hat, k=k) for k in range(1, M + s)],
            dtype=float,
        )
    )

    beta_j = np.concatenate([[1.0], beta_offdiag])
    C_matrix = np.eye(M + s)
    for i in range(s):
        sl = beta_offdiag if i == 0 else beta_offdiag[:-i]
        C_matrix[i, 1 + i :] = sl

    try:
        D_matrix = np.linalg.inv(C_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("C_matrix is singular.")

    if isinstance(y, list):
        y_arr = np.asarray(y, dtype=float).ravel()
        v_filt = y_arr.copy()
        if np.any(np.abs(phi_hat) > 1e-10):
            for i in range(1, len(phi_hat) + 1):
                v_filt[i:] -= phi_hat[i - 1] * y_arr[:-i]
        v_vector = np.array([v_filt[-s + i] for i in range(s)]).reshape(-1, 1)
    else:
        v_vector = y

    first_row = np.full((s, N_draws), v_vector)
    w_matrix = np.vstack([first_row, eps])
    e_matrix = D_matrix @ w_matrix
    e_col = e_matrix[:s, :].T

    # 3) Importance weights w_i = prod_{j=1}^s f_sigma(e_{T-s+j})
    f_sigma = stable_pdf(e_col, alpha_hat, beta_hat, loc=0, scale=sigma_hat)
    f_sigma = np.reshape(f_sigma, e_col.shape)  # (N_draws, s)
    weights = np.prod(f_sigma, axis=1)  # (N_draws,)
    den = weights.sum()

    # 4) Compute v_{T+h} with AR dynamics (if applicable)
    if r > 0 and np.any(np.abs(phi_hat) > 1e-10):
        y_vec = np.array(
            [y[-i] for i in range(1, r + 1)]
        )  # [y_T, y_{T-1}, ..., y_{T-r+1}]

        # Build companion matrix: Φ
        Phi = np.zeros((r, r))
        Phi[0, :] = phi_hat
        if r > 1:
            Phi[1:, :-1] = np.eye(r - 1)

        # Compute deterministic part: e₁'Φ^h @ y_vec
        Phi_h = np.linalg.matrix_power(Phi, horizon)
        e1 = np.zeros(r)
        e1[0] = 1.0
        deterministic_part = e1 @ Phi_h @ y_vec

        # Compute stochastic part:
        iota = np.zeros(r)
        iota[0] = 1.0

        v_T_plus_h = np.zeros((1, N_draws))
        for i in range(horizon):
            # Compute scalar coefficient: ι'Φⁱι
            if i == 0:
                coef = 1.0
            else:
                Phi_i = np.linalg.matrix_power(Phi, i)
                coef = iota @ Phi_i @ iota

            # Add: coef * Σⱼ₌₀^{M-h+i} βⱼε_{T+h-i+j}
            # eps[h-i-1:M, :] contains ε_{T+h-i}, ..., ε_{T+M}
            n_terms = M - horizon + i + 1
            v_T_plus_h += coef * beta_j[None, :n_terms] @ eps[horizon - i - 1 :, :]

        # Shift grid by deterministic part
        grid_y_shifted = grid_y - deterministic_part
    else:
        # Purely noncausal case (r=0): v_{T+h} ≈ Σⱼ₌₀^{M-h} βⱼε_{T+h+j}
        v_T_plus_h = beta_j[None, : M - horizon + 1] @ eps[horizon - 1 :, :]
        grid_y_shifted = grid_y  # No deterministic shift

    v_T_plus_h_col = v_T_plus_h.T

    # 5) Compute the weighted CDF on (shifted) grid_y
    boolean_num = v_T_plus_h_col <= grid_y_shifted[None, :]  # (N_draws, len(grid_y))
    cdf_hat = (boolean_num.T @ weights) / den  # (len(grid_y),)
    cdf_hat = cdf_hat[:, None]

    # 6) Numerical derivative → predictive density
    pdf_hat = np.gradient(cdf_hat[:, 0], grid_y)
    pdf_hat /= np.trapezoid(pdf_hat, grid_y)

    return pdf_hat, cdf_hat


def lls2012_predictive_density(
    psi_hat,
    alpha_hat,
    beta_hat,
    sigma_hat,
    horizon,
    M,
    N_draws,
    grid_x,
    grid_y,
    n_process,
    phi_hat=None,
    seed=None,
):
    """Predictive densities over (grid_x, grid_y), computed in parallel.

    ``M`` truncates the simulation horizon and ``N_draws`` sets the sample size.
    """
    if phi_hat is None:
        phi_hat = [0.0]

    def utils(y_t, row_seed):
        pdf_hat, _ = lls2012_predictive_density_given_yt(
            psi_hat=psi_hat,
            phi_hat=phi_hat,
            alpha_hat=alpha_hat,
            beta_hat=beta_hat,
            sigma_hat=sigma_hat,
            y=y_t,
            horizon=horizon,
            M=M,
            N_draws=N_draws,
            grid_y=grid_y,
            seed=row_seed,
        )
        return pdf_hat  # shape: len(grid_y,)

    def _row_seed(i):
        return None if seed is None else int((seed + i) % (2**31 - 1))

    with parallel_backend("loky"):
        rows = Parallel(n_jobs=n_process)(
            delayed(utils)(float(y_t), _row_seed(i))
            for i, y_t in enumerate(tqdm(grid_x, desc="rows"))
        )

    return np.vstack(rows)  # shape: len(grid_x) x len(grid_y)
