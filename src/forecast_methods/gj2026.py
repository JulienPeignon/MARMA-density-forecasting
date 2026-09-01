"""Gourieroux-Jasiak (2026) noncausal predictive density."""

import numpy as np
from scipy.linalg import schur
from sklearn.neighbors import KernelDensity

from src.forecast_methods.utils import compute_mode
from src.stable_mar.stable import stable_pdf


def gj2026_predictive_density_given_yt(
    psi_hat: list,
    phi_hat: list,
    alpha_hat: float,
    beta_hat: float,
    sigma_hat: float,
    y_train: np.ndarray,
    y_history: list,
    grid_y: np.ndarray,
    bandwidth: str = "silverman",
    horizon: int = 1,
):
    """Gourieroux-Jasiak (2026) predictive density for a MAR(r, s)."""
    s = len(psi_hat)
    r = 0 if phi_hat == [0.0] else len(phi_hat)

    current_density = _compute_h1_density(
        psi_hat,
        phi_hat,
        alpha_hat,
        beta_hat,
        sigma_hat,
        y_train,
        y_history,
        grid_y,
        bandwidth,
        s,
        r,
    )

    if horizon == 1:
        return current_density

    # Multi-horizon: iterate using mode-based prediction
    for h in range(2, horizon + 1):
        mode = compute_mode(current_density, grid_y)

        # Update history: prepend the mode, keep the p = r+s most recent values.
        new_history = ([mode] + list(y_history))[: r + s]

        current_density = _compute_h1_density(
            psi_hat,
            phi_hat,
            alpha_hat,
            beta_hat,
            sigma_hat,
            y_train,
            new_history,
            grid_y,
            bandwidth,
            s,
            r,
        )

    return current_density


def compute_jordan_decomposition(Phi):
    """Jordan form ``Phi = A @ diag(J1, J2) @ A^-1``.

    J1 holds the stable eigenvalues (|lambda| < 1), J2 the unstable ones.
    Returns (A, J1, J2, A_inv).
    """
    eigenvalues = np.linalg.eigvals(Phi)

    # If complex roots, use real Schur decomposition
    if not np.allclose(eigenvalues.imag, 0):
        T, Z = schur(Phi, output="real")
        n_stable = np.sum(np.abs(eigenvalues) < 1)

        A = Z
        A_inv = Z.T
        J1 = T[:n_stable, :n_stable] if n_stable > 0 else None
        J2 = T[n_stable:, n_stable:] if n_stable < len(eigenvalues) else None

        return A, J1, J2, A_inv, n_stable

    # Real eigenvalues
    eigenvectors = np.linalg.eig(Phi)[1]

    stable_mask = np.abs(eigenvalues) < 1
    stable_indices = np.where(stable_mask)[0]
    unstable_indices = np.where(~stable_mask)[0]

    reordered_indices = np.concatenate([stable_indices, unstable_indices])

    A = np.real(eigenvectors[:, reordered_indices])
    n_stable = len(stable_indices)

    J1 = np.real(np.diag(eigenvalues[stable_indices])) if n_stable > 0 else None
    J2 = (
        np.real(np.diag(eigenvalues[unstable_indices]))
        if len(unstable_indices) > 0
        else None
    )

    A_inv = np.linalg.inv(A)

    return A, J1, J2, A_inv, n_stable


def _compute_h1_density(
    psi_hat,
    phi_hat,
    alpha_hat,
    beta_hat,
    sigma_hat,
    y_train,
    y_history,
    grid_y,
    bandwidth,
    s,
    r,
):
    """Compute the 1-step-ahead predictive density (GJ 2026, eq. 2.1)."""
    return _compute_h1_general(
        psi_hat,
        phi_hat,
        alpha_hat,
        beta_hat,
        sigma_hat,
        y_train,
        y_history,
        grid_y,
        bandwidth,
        s,
        r,
    )


def _reduced_ar_coefficients(psi, phi, s, r):
    """Reduced AR(p=r+s) coefficients of the mVAR(1.1) representation of MAR(r,s)."""
    lam = (
        np.roots(np.concatenate(([1.0], -np.asarray(phi, dtype=float))))
        if r > 0
        else np.array([], dtype=complex)
    )  # causal companion eigenvalues, |lam| < 1
    kappa = np.roots(
        np.concatenate(((-np.asarray(psi, dtype=float))[::-1], [1.0]))
    )  # roots of 1 - psi_1 z - ... - psi_s z^s, |kappa| > 1
    eigs = np.concatenate([lam, kappa])
    return -np.real(np.poly(eigs)[1:])  # Phi_1 .. Phi_p


def _compute_h1_general(
    psi_hat,
    phi_hat,
    alpha_hat,
    beta_hat,
    sigma_hat,
    y_train,
    y_history,
    grid_y,
    bandwidth,
    s,
    r,
):
    """GJ (2026) eq. (2.1) predictive density for a general MAR(r, s)."""
    y_train = np.asarray(y_train, dtype=float).ravel()
    grid_y = np.asarray(grid_y, dtype=float).ravel()
    psi = np.asarray(psi_hat, dtype=float)
    phi = np.zeros(0) if r == 0 else np.asarray(phi_hat, dtype=float)
    p = r + s

    y_hist = np.array(
        [float(np.ravel(v)[0]) for v in y_history[:p]]
    )  # [y_T..y_{T-p+1}]

    # Reduced AR(p) form and its companion matrix.
    coefs = _reduced_ar_coefficients(psi, phi, s, r)
    Phi = np.zeros((p, p))
    Phi[0, :] = coefs
    if p > 1:
        Phi[1:, :-1] = np.eye(p - 1)

    _, _, _, A_inv, n_stable = compute_jordan_decomposition(Phi)
    A2 = A_inv[n_stable:, :]  # (n2 x p): rows for the noncausal state Z_2

    n_obs = len(y_train)
    windows = np.array([y_train[t - p + 1 : t + 1][::-1] for t in range(p - 1, n_obs)])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        states = windows @ A2.T
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(states)

    # Numerator states: [y, y_T, ..., y_{T-p+2}] for each candidate y on the grid.
    num_windows = np.empty((grid_y.size, p))
    num_windows[:, 0] = grid_y
    if p > 1:
        num_windows[:, 1:] = y_hist[: p - 1]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        log_l2_num = kde.score_samples(num_windows @ A2.T)
    density_ratio = np.exp(log_l2_num - log_l2_num.max())

    # Reduced-form residual and transformed alpha-stable error density g.
    epsilon = grid_y - float(np.dot(coefs, y_hist))
    coef = -1.0 / psi[s - 1]
    g_eps = stable_pdf(
        epsilon,
        alpha_hat,
        np.sign(coef) * beta_hat,
        loc=0,
        scale=np.abs(coef) * sigma_hat,
    )

    eig = np.linalg.eigvals(Phi)
    jacobian = float(np.prod(np.abs(eig[np.abs(eig) > 1])))  # |det J_2|

    pdf = density_ratio * jacobian * g_eps
    pdf /= np.trapezoid(pdf, grid_y)

    return pdf
