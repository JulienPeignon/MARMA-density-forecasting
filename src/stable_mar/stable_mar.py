"""Alpha-stable MAR/MARMA simulation, estimation and inference."""

import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.signal import lfilter

from src.stable_mar.stable import stable_pdf, stable_rvs


class stablemar:
    """MAR(r, s) model with alpha-stable innovations.

    Psi(F) Phi(B) X_t = eps_t, with F the forward and B the backward operator,
    and eps_t ~ S(alpha, beta, sigma, 0).
    """

    def __init__(self, order: Tuple[int, int], par: List[float] = None):
        """Set the model order and, optionally, its parameters."""
        self.order = order
        self.par = par if par is not None else []
        self.results = {}
        self.trajectory = None
        self.innovation = None

    def fit(
        self,
        data: np.ndarray,
        start: List[float],
        method: str = "gcov",
        max_lag: int = 10,
        K: int = 2,
        H: int = 2,
        verbose: bool = False,
    ) -> "stablemar":
        """Estimate the model by "gcov", "mdsd" or "mle".

        For "mle", ``start`` is the constrained layout (phi_c, phi_nc, alpha, beta,
        sigma); None computes McCulloch + OLS starting values.
        """
        if method == "gcov":
            return self._fit_gcov(data, start, K, H, verbose=verbose)
        elif method == "mdsd":
            return self._fit_mdsd(data, start, max_lag, verbose=verbose)
        elif method == "mle":
            return self._fit_mle(data, start, verbose=verbose)
        else:
            raise ValueError("Method must be one of 'gcov', 'mdsd', or 'mle'")

    def _fit_mle(
        self,
        data: np.ndarray,
        start: List[float] = None,
        verbose: bool = False,
    ) -> "stablemar":
        """Estimate the MAR by profile MLE with alpha-stable innovations.

        Port of ``marx.estim.alpha_sc()`` from estimation_functions_MLE.R. mu is
        concentrated out (mu_hat = mean(E)) and the density evaluated on the centred
        innovations; L-BFGS-B runs on the constrained space with box bounds.
        """
        r, s = self.order
        params0 = None if (start is None or len(start) == 0) else start
        res = _mle_marx_estim(data, r, s, params0)
        params = res["params"]

        self.par = [float(v) for v in params]
        self.results = {
            "Parameters": np.asarray(params[: r + s], dtype=float),
            "StableParams": [float(v) for v in params[r + s : r + s + 3]],
            "mu_hat": float(res["mu_hat"]),
            "PseudoResiduals": np.asarray(res["E"], dtype=float),
            "Loglik": float(res["Loglik"]),
            "Method": "mle",
        }

        if verbose:
            print(
                f"MLE MAR({r},{s}) | Loglik = {-res['Loglik']:.4f} | "
                f"params = {self.par}"
            )

        return self

    def _fit_gcov(
        self,
        data: np.ndarray,
        start: List[float],
        K: int = 2,
        H: int = 2,
        verbose: bool = False,
    ) -> "stablemar":
        """Estimate the MAR by generalized covariance."""
        r, s = self.order
        gcov_start = start[: r + s]

        def objective_function(theta: np.ndarray) -> float:
            return self.gcov(data, theta, K, H)[0]

        bounds = self._param_bounds(r + s)

        if verbose:
            print(f"Starting gcov estimation with K={K}, H={H}")
            print(f"Initial parameters: {gcov_start}")
            start_time = time.time()

        optimum = minimize(
            objective_function, gcov_start, bounds=bounds, method="L-BFGS-B"
        )

        if verbose:
            elapsed_time = time.time() - start_time
            print(f"Optimization completed in {elapsed_time:.2f} seconds")
            print(f"Final parameters: {optimum.x}")

        mar_params = optimum.x
        loss_stat, pseudo_residuals = self.gcov(data, mar_params, K, H)

        self.results = {
            "Parameters": mar_params,
            "GStatistic": loss_stat,
            "PseudoResiduals": pseudo_residuals,
            "Method": "gcov",
        }

        self.par = mar_params.tolist()

        return self

    def _fit_mdsd(
        self,
        data: np.ndarray,
        start: List[float],
        max_lag: int = 10,
        verbose: bool = False,
    ) -> "stablemar":
        """Estimate the MAR by minimum distance on the spectral density.

        Grid-searched; see Velasco (2022).
        """
        r, s = self.order
        mdsd_start = start[: r + s]

        if verbose:
            print(f"Starting mdsd estimation for MAR({r},{s}) model")
            print(f"Initial parameters: {mdsd_start}")
            start_time = time.time()

        def objective_function(theta: np.ndarray) -> float:
            return self._mds_criterion(data, theta, max_lag, method="L")

        bounds = self._param_bounds(r + s)

        try:
            optimum = minimize(
                objective_function,
                mdsd_start,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": 200},
            )

            mar_params = optimum.x
            criterion_value = optimum.fun
            success = optimum.success

            if not success and verbose:
                print(f"Warning: Optimization did not converge: {optimum.message}")
        except Exception as e:
            if verbose:
                print(f"Optimization failed: {str(e)}")
                print("Using initial guess parameters as fallback")

            mar_params = mdsd_start
            criterion_value = objective_function(mdsd_start)

        residuals, std_residuals = self._pseudo_residuals(data, mar_params)

        self.results = {
            "Parameters": mar_params,
            "CriterionValue": criterion_value,
            "Residuals": residuals,
            "StdResiduals": std_residuals,
            "Method": "mdsd",
            "InitialGuess": mdsd_start,
        }

        self.par = mar_params.tolist()

        if verbose:
            elapsed_time = time.time() - start_time
            print(f"MDSD estimation completed in {elapsed_time:.2f} seconds")
            print(f"Final parameters: {self.par}")
            print(f"Final criterion value: {criterion_value}")

        return self

    def _mds_criterion(
        self,
        data: np.ndarray,
        params: np.ndarray,
        max_lag: int = 10,
        method: str = "Q",
        sigma: float = 1.0,
    ) -> float:
        """Minimum-distance MDSD criterion; ``method`` is "L" or "Q"."""
        _, residuals = self._pseudo_residuals(data, params)

        lags = range(1, min(max_lag + 1, len(residuals)))

        sigma_hat = self._compute_cf_covariances(residuals, lags, sigma)

        if method == "L":
            criterion = 0
            for j in lags:
                criterion += (1 / (j**2)) * sigma_hat.get(j, 0)
            criterion *= 2 / np.pi
        elif method == "Q":  # method == 'Q'
            T = len(residuals)

            # Daniell kernel
            def kernel(x: float) -> float:
                return np.sin(np.pi * x) / (np.pi * x) if x != 0 else 1

            p = int(T ** (1 / 5))

            criterion = 0
            for j in lags:
                k_j = kernel(j / p)
                correction = 1 - (j / T)
                criterion += (k_j**2) * correction * sigma_hat.get(j, 0)
            criterion *= 2 / np.pi
        else:
            print("Unknown method, should be L or Q")

        return criterion

    def _compute_cf_covariances(
        self, residuals: np.ndarray, lags: range, sigma: float = 1.0
    ) -> Dict[int, float]:
        """Characteristic-function covariances, by lag."""
        v_grid = np.linspace(-2, 2, 20) * sigma

        sigma_hat = {}

        exp_iv_e = np.array([np.exp(1j * v * residuals) for v in v_grid])
        phi_v = np.mean(exp_iv_e, axis=1)

        for j in lags:
            if j == 0:
                continue

            sigma_j_values = []

            for idx, v in enumerate(v_grid):
                exp_iv_e_lag = exp_iv_e[idx, :-j]  # e^{iv*e_{t-j}}
                i_e = 1j * residuals[j:]  # i*e_t

                # Calculate (exp(i*v*e_{t-j}) - E[exp(i*v*e)])
                centered_exp = exp_iv_e_lag - phi_v[idx]

                # Calculate i*e_t * (exp(i*v*e_{t-j}) - E[exp(i*v*e)])
                term = i_e * centered_exp

                sigma_j_v = np.abs(np.mean(term)) ** 2
                sigma_j_values.append(sigma_j_v)

            # Integrate over v (numerical approximation)
            sigma_hat[j] = np.mean(sigma_j_values)

        return sigma_hat

    def _pseudo_residuals(
        self, data: np.ndarray, params: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the pseudo residuals and their standardised form."""
        r, s = self.order
        n = len(data)

        psi = params[:r] if r > 0 else np.array([])
        phi = params[r : r + s] if s > 0 else np.array([])

        pseudo = np.zeros(n)

        for t in range(r, n - s):
            pseudo[t] = data[t]

            for i in range(r):
                pseudo[t] -= psi[i] * data[t - i - 1]

            for j in range(s):
                pseudo[t] -= phi[j] * data[t + j + 1]

            for i in range(r):
                for j in range(s):
                    pseudo[t] += psi[i] * phi[j] * data[t - i + j]

        stdpseudo = (pseudo - np.mean(pseudo)) / np.std(pseudo)

        return pseudo[r : n - s], stdpseudo[r : n - s]

    def gcov(
        self, x: np.ndarray, theta: np.ndarray, K: int, H: int
    ) -> Tuple[float, np.ndarray]:
        """Compute the generalized covariance criterion of the MAR.

        See Gourieroux and Jasiak (2017, 2023). Returns the loss and the residuals.
        """
        pseudo, _ = self._pseudo_residuals(x, theta)

        kpowers = np.arange(1, K + 1)

        eps = pseudo[:, np.newaxis] ** kpowers

        eps = eps - np.mean(eps, axis=0)

        # 1/Gamma(0) computation (variance-covariance matrix)
        n_eps = eps.shape[0]
        vcv = np.dot(eps.T, eps) / n_eps

        try:
            ivcv = np.linalg.inv(vcv)
        except np.linalg.LinAlgError:
            # pseudo-inverse when singular
            ivcv = np.linalg.pinv(vcv)
            warnings.warn("Singular matrix in gcov estimation, using pseudo-inverse")

        # Gamma(h) computation (autocovariance matrices at different lags)
        vcvmat = np.array(
            [np.dot(eps[i:].T, eps[:-i]) / (n_eps - i) for i in range(1, H + 1)]
        )

        # Loss statistic L(theta) = sum(Tr(R^2)), with R^2 = cov * invv * cov.T * invv
        ls = np.sum(
            [np.trace(np.linalg.multi_dot([h, ivcv, h.T, ivcv])) for h in vcvmat]
        )

        return ls, pseudo

    def fit_stable_noise(
        self, data: np.ndarray, step: float = 0.01, m: float = 1.0
    ) -> List[float]:
        """Estimate alpha-stable parameters from the residuals.

        Uses the characteristic function of Kogon and Williams (1998).
        """
        n = len(data)
        u = np.arange(step, m + step, step)

        # Robust scale estimation using IQR
        q75 = np.quantile(data, 0.75)
        q25 = np.quantile(data, 0.25)
        iqr = q75 - q25
        sigma0 = iqr / 2

        data = data / sigma0

        ecf1 = np.array([(1 / n) * np.sum(np.cos(ui * data)) for ui in u])
        ecf2 = np.array([(1 / n) * np.sum(np.sin(ui * data)) for ui in u])
        ecf = ecf1 + 1j * ecf2

        # Estimate alpha and sigma from the log-log plot
        x = np.log(-np.log(np.abs(ecf) + 1e-10))
        y = np.log(np.abs(u) + 1e-10)
        parhat = np.polyfit(y, x, 1)

        alpha = parhat[0]

        # Constrain alpha to be in (0, 2)
        alpha = max(0.01, min(1.99, alpha))

        sigma = np.exp(parhat[1] / alpha)

        if alpha != 1:
            eta = np.tan(np.pi * alpha / 2) * (np.abs(u) - np.abs(u) ** alpha)
        else:
            eta = 2 / np.pi * u * np.log(np.abs(u) + 1e-10)

        x = np.arctan2(ecf2, ecf1)
        y = -(sigma**alpha) * eta

        parhat = np.polyfit(x, y, 1)
        beta = parhat[0]

        # Constrain beta to be in [-1, 1]
        beta = max(-0.99, min(0.99, beta))

        sigma = sigma0 * sigma

        return [float(alpha), float(beta), float(sigma)]

    def inference(
        self,
        data: np.ndarray,
        params: Optional[List[float]] = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Estimates, standard errors, t-statistics, p-values and intervals."""
        if params is None:
            if self.par is None:
                raise ValueError(
                    "Parameters must be estimated before running inference"
                )
            params = self.par

        r, s = self.order
        params_array = np.array(params)

        method = self.results.get("Method", "Unknown")

        param_names = []

        for i in range(r):
            param_names.append(f"phi_{i + 1}")

        for i in range(s):
            param_names.append(f"psi_{i + 1}")

        if len(params) > r + s:
            param_names.append("alpha")
            if len(params) > r + s + 1:
                param_names.append("beta")
                if len(params) > r + s + 2:
                    param_names.append("sigma")

        vcv_matrix = None

        try:
            if method == "gcov":
                vcv_matrix = self._calculate_gcov_vcv(data, params)
            elif method == "mdsd":
                vcv_matrix = self._calculate_mdsd_vcv(data, params)
            elif method == "mle":
                # Observed information: inverse numerical Hessian of the NLL.
                vcv_matrix = self._calculate_mle_vcv(data, params)
            else:
                print(
                    f"Warning: unknown estimation method '{method}'; "
                    f"falling back to approximations."
                )
                vcv_matrix = np.diag([0.1] * len(params))
        except Exception as e:
            print(f"Error computing the VCV matrix: {str(e)}")
            print("Falling back to an approximate VCV matrix")
            vcv_matrix = np.diag([0.1] * len(params))

        std_errors = np.sqrt(np.diag(vcv_matrix))

        t_stats = params_array[: len(param_names)] / std_errors

        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

        z_critical = stats.norm.ppf(1 - alpha / 2)
        lower_ci = params_array[: len(param_names)] - z_critical * std_errors
        upper_ci = params_array[: len(param_names)] + z_critical * std_errors

        results = pd.DataFrame(
            {
                "Parameter": param_names,
                "Estimate": params_array[: len(param_names)],
                "Std. Error": std_errors,
                "t-statistic": t_stats,
                "p-value": p_values,
                f"CI {100 * (1 - alpha)}% Lower": lower_ci,
                f"CI {100 * (1 - alpha)}% Upper": upper_ci,
            }
        )

        return results

    def _calculate_mle_vcv(self, data: np.ndarray, params: List[float]) -> np.ndarray:
        """Variance-covariance matrix for the MLE."""
        r, s = self.order
        p_C, p_NC = r, s
        y = np.asarray(data, dtype=float).ravel()
        params_mle = np.asarray(params, dtype=float)
        d = len(params_mle)
        lower, upper = _mle_make_bounds(p_C, p_NC)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        h = 1e-5

        def nll(v):
            return _mle_nll_mar_mu0(np.clip(v, lower, upper), y, p_C, p_NC)

        def unit(i):
            e = np.zeros(d)
            e[i] = h
            return e

        fx = nll(params_mle)

        # Numerical Hessian (same finite-difference formula as the R code).
        grdp = np.array([nll(params_mle + unit(i)) for i in range(d)])
        grdm = np.array([nll(params_mle - unit(i)) for i in range(d)])
        hess = np.full((d, d), np.nan)
        for i in range(d):
            for j in range(i, d):
                fpp = nll(params_mle + unit(i) + unit(j))
                fmm = nll(params_mle - unit(i) - unit(j))
                hess[i, j] = hess[j, i] = (
                    fpp - grdp[i] - grdp[j] + fx + fx - grdm[i] - grdm[j] + fmm
                ) / (2.0 * h**2)

        try:
            hinv = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            hinv = np.linalg.pinv(hess)

        return hinv

    def _calculate_gcov_vcv(self, data: np.ndarray, params: List[float]) -> np.ndarray:
        """Variance-covariance matrix of the GCov estimator.

        Follows the asymptotic theory of the GCOVJBES paper.
        """
        r, s = self.order
        n = len(data)

        residuals = self.results.get("PseudoResiduals", None)
        if residuals is None:
            residuals, _ = self._pseudo_residuals(data, np.array(params))

        p = r + s

        # Regle empirique : racine carree de la precision machine,
        # multipliee par un facteur d'echelle
        epsilon = max(1e-4, np.sqrt(np.finfo(float).eps) * 0.1)

        Theta = np.zeros((p, p))

        gamma_0 = max(np.var(residuals), 1e-6)  # Éviter division par zéro

        for i in range(p):
            for j in range(p):
                # relative perturbation
                delta_i = max(epsilon, abs(params[i] * 0.01))
                delta_j = max(epsilon, abs(params[j] * 0.01))

                params_plus_i = params.copy()
                params_minus_i = params.copy()
                params_plus_i[i] += delta_i
                params_minus_i[i] -= delta_i

                params_plus_j = params.copy()
                params_minus_j = params.copy()
                params_plus_j[j] += delta_j
                params_minus_j[j] -= delta_j

                # Derivees croisees des autocovariances, H premiers lags
                H = min(10, n // 4)  # Limiter le nombre de lags pour la stabilité
                derivative_sum = 0

                for h in range(1, H + 1):
                    gamma_h_plus_i = self._compute_acf(data, params_plus_i, h)
                    gamma_h_minus_i = self._compute_acf(data, params_minus_i, h)
                    d_gamma_h_d_theta_i = (gamma_h_plus_i - gamma_h_minus_i) / (
                        2 * delta_i
                    )

                    gamma_h_plus_j = self._compute_acf(data, params_plus_j, h)
                    gamma_h_minus_j = self._compute_acf(data, params_minus_j, h)
                    d_gamma_h_d_theta_j = (gamma_h_plus_j - gamma_h_minus_j) / (
                        2 * delta_j
                    )

                    # Proposition 4
                    derivative_sum += d_gamma_h_d_theta_i * d_gamma_h_d_theta_j

                # Corollary 1: normalise by gamma_0^2
                Theta[i, j] = derivative_sum / (gamma_0**2)

        # jitter to keep Theta positive definite
        Theta += np.eye(p) * max(1e-6, 0.001 * np.trace(Theta) / p)

        # check the conditioning
        try:
            cond_num = np.linalg.cond(Theta)
            if cond_num > 1e10:  # Mauvais conditionnement
                print(f"Warning: Theta matrix ill-conditioned (cond = {cond_num:.2e})")
                # stronger regularisation
                Theta += np.eye(p) * max(1e-4, 0.01 * np.trace(Theta) / p)
        except Exception:
            print("Warning: Failed to solve Theta matrix ill-conditioned issue")
            # default regularisation
            Theta += np.eye(p) * 0.01

        try:
            vcv_matrix = np.linalg.inv(Theta) / n
        except np.linalg.LinAlgError:
            print("Warning: Failed to invert Theta, using pseudo-inverse.")
            vcv_matrix = np.linalg.pinv(Theta) / n

        for i in range(p):
            if (
                vcv_matrix[i, i] <= 0
                or np.isnan(vcv_matrix[i, i])
                or np.isinf(vcv_matrix[i, i])
            ):
                print(f"Warning: Negative or invalid variance for parameter {i}")
                vcv_matrix[i, i] = 0.01

        # extend to the stable parameters, when present
        if len(params) > p:
            vcv_stable = self._calculate_stable_params_vcv(residuals, params[p:])
            vcv_matrix_extended = np.zeros((len(params), len(params)))

            vcv_matrix_extended[:p, :p] = vcv_matrix
            vcv_matrix_extended[p:, p:] = vcv_stable

            return vcv_matrix_extended

        return vcv_matrix

    def _calculate_mdsd_vcv(self, data: np.ndarray, params: List[float]) -> np.ndarray:
        """Variance-covariance matrix of the MDSD estimator.

        Follows Velasco (2022).
        """
        r, s = self.order
        n = len(data)

        residuals = self.results.get("Residuals", None)
        if residuals is None:
            residuals, _ = self._pseudo_residuals(data, np.array(params))

        p = r + s

        epsilon = max(1e-4, np.sqrt(np.finfo(float).eps) * 0.1)

        gradient = np.zeros((n, p))

        for j in range(p):
            delta_j = max(epsilon, abs(params[j] * 0.01))

            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[j] += delta_j
            params_minus[j] -= delta_j

            try:
                criterion_plus = self._mds_criterion(data, params_plus, method="L")
                criterion_minus = self._mds_criterion(data, params_minus, method="L")
                gradient[:, j] = (criterion_plus - criterion_minus) / (2 * delta_j)
            except Exception as e:
                print(f"Error computing the gradient for parameter {j}: {str(e)}")
                gradient[:, j] = np.random.normal(0, 0.01, n)

        hessian = np.zeros((p, p))

        for i in range(p):
            for j in range(p):
                delta_i = max(epsilon, abs(params[i] * 0.01))
                delta_j = max(epsilon, abs(params[j] * 0.01))

                try:
                    params_pp = params.copy()
                    params_pm = params.copy()
                    params_mp = params.copy()
                    params_mm = params.copy()

                    params_pp[i] += delta_i
                    params_pp[j] += delta_j

                    params_pm[i] += delta_i
                    params_pm[j] -= delta_j

                    params_mp[i] -= delta_i
                    params_mp[j] += delta_j

                    params_mm[i] -= delta_i
                    params_mm[j] -= delta_j

                    criterion_pp = self._mds_criterion(data, params_pp, method="L")
                    criterion_pm = self._mds_criterion(data, params_pm, method="L")
                    criterion_mp = self._mds_criterion(data, params_mp, method="L")
                    criterion_mm = self._mds_criterion(data, params_mm, method="L")

                    hessian[i, j] = (
                        criterion_pp - criterion_pm - criterion_mp + criterion_mm
                    ) / (4 * delta_i * delta_j)
                except Exception as e:
                    print(
                        f"Error computing the Hessian for parameters {i},{j}: {str(e)}"
                    )
                    if i == j:
                        hessian[i, j] = 0.1  # Valeur positive pour la diagonale
                    else:
                        hessian[i, j] = 0.0

        # force the Hessian positive definite
        hessian_diag = np.diag(hessian)
        if np.any(hessian_diag <= 0):
            print("Warning: Hessian not positive definite; adding regularisation.")
            min_diag = max(0.1, np.median(np.abs(hessian_diag)) * 0.01)
            for i in range(p):
                if hessian[i, i] <= 0:
                    hessian[i, i] = min_diag

        score_cov = np.zeros((p, p))

        # assuming the scores are mds errors
        for i in range(p):
            for j in range(p):
                for t in range(n):
                    score_cov[i, j] += gradient[t, i] * gradient[t, j]

        score_cov /= n

        min_eig = np.min(np.linalg.eigvals(score_cov))
        if min_eig < 1e-10:
            print(
                f"Warning: ill-conditioned score covariance matrix "
                f"(min_eig = {min_eig:.2e})"
            )
            score_cov += np.eye(p) * max(1e-4, 0.01 * np.trace(score_cov) / p)

        try:
            inv_hessian = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            print("Warning: singular Hessian; applying stronger regularisation.")
            hessian_reg = hessian + 0.01 * np.eye(p) * np.mean(np.abs(np.diag(hessian)))
            inv_hessian = np.linalg.inv(hessian_reg)

        # sandwich formula
        vcv_matrix = inv_hessian @ score_cov @ inv_hessian / n

        for i in range(p):
            if (
                vcv_matrix[i, i] <= 0
                or np.isnan(vcv_matrix[i, i])
                or np.isinf(vcv_matrix[i, i])
            ):
                print(f"Warning: negative or invalid variance for parameter {i}")
                # default from the other elements
                abs_diag = np.abs(np.diag(vcv_matrix))
                abs_diag = abs_diag[
                    ~np.isnan(abs_diag) & ~np.isinf(abs_diag) & (abs_diag > 0)
                ]
                if len(abs_diag) > 0:
                    vcv_matrix[i, i] = np.median(abs_diag)
                else:
                    vcv_matrix[i, i] = 0.01

        # extend to the stable parameters, when present
        if len(params) > p:
            vcv_stable = self._calculate_stable_params_vcv(residuals, params[p:])
            vcv_matrix_extended = np.zeros((len(params), len(params)))

            vcv_matrix_extended[:p, :p] = vcv_matrix
            vcv_matrix_extended[p:, p:] = vcv_stable

            return vcv_matrix_extended

        return vcv_matrix

    def _calculate_stable_params_vcv(
        self, residuals: np.ndarray, stable_params: List[float]
    ) -> np.ndarray:
        """Variance-covariance matrix of the stable parameters."""
        n_stable = len(stable_params)
        n = len(residuals)

        # typical empirical variances
        vcv_stable = np.zeros((n_stable, n_stable))

        if n_stable >= 1:
            alpha = stable_params[0]
            # var(alpha) grows as alpha approaches 2
            alpha_var = 0.02 + 0.03 * (alpha / 2.0) ** 2
            vcv_stable[0, 0] = alpha_var

        if n_stable >= 2:
            beta = stable_params[1]
            # var(beta) grows as |beta| approaches 1
            beta_var = 0.04 + 0.06 * beta**2
            vcv_stable[1, 1] = beta_var

            # alpha-beta covariance, usually small
            vcv_stable[0, 1] = vcv_stable[1, 0] = 0.01

        if n_stable >= 3:
            sigma = stable_params[2]
            # var(sigma) is proportional to sigma^2
            sigma_var = 0.01 * sigma**2
            vcv_stable[2, 2] = sigma_var

            # negligible cross-covariances
            vcv_stable[0, 2] = vcv_stable[2, 0] = 0.005
            vcv_stable[1, 2] = vcv_stable[2, 1] = 0.005

        # sample-size adjustment
        vcv_stable = vcv_stable * (200 / n) if n > 0 else vcv_stable

        return vcv_stable

    def _compute_acf(self, data: np.ndarray, params: List[float], lag: int) -> float:
        """Residual autocovariance at ``lag`` for ``params``."""
        try:
            residuals, _ = self._pseudo_residuals(data, np.array(params))

            n = len(residuals)
            if n <= lag:
                return 0.0

            mean = np.mean(residuals)
            acf = 0

            for t in range(lag, n):
                acf += (residuals[t] - mean) * (residuals[t - lag] - mean)

            return acf / (n - lag)
        except Exception as e:
            print(f"Error computing the ACF at lag {lag}: {str(e)}")
            return 0.0

    def generate(
        self, n: int, errors: List[float] = None, seed: Optional[int] = None
    ) -> "stablemar":
        """Simulate the MAR(r, s), setting the trajectory and innovations."""
        if self.par is None or len(self.par) < 3:
            raise ValueError("Parameters must be set before generating data")

        r, s = self.order

        if len(self.par) > r + s:
            alpha = self.par[r + s]
            beta = self.par[r + s + 1] if len(self.par) > r + s + 1 else 0.0
            sigma = self.par[r + s + 2] if len(self.par) > r + s + 2 else 1.0
        else:
            alpha, beta, sigma = 1.5, 0.0, 1.0  # Default values

        if seed is not None:
            np.random.seed(seed)

        m = 50  # Truncation for the MA filter
        if n < 2 * m:
            warnings.warn(f"Sample size (n={n}) is too small... n >= 100 is required")

        ntilde = n + 2 * m + 1

        deltas = self.ma_filter(m)
        deltas = np.flip(deltas)  # Respecter l'orientation de l'original

        if errors is None or len(errors) != ntilde:
            esim = stable_rvs(alpha=alpha, beta=beta, scale=sigma, loc=0, size=n * 3)
        else:
            esim = errors

        xsim = np.ones(ntilde) * esim[0]

        for t in range(m, ntilde):
            xsim[t] = np.sum(deltas * esim[t - m : t + m])

        xsim = xsim[m:-m]
        esim = esim[m:-m]

        self.trajectory = pd.Series(xsim)
        self.innovation = pd.Series(esim)

        return self

    def ma_filter(self, m: int) -> np.ndarray:
        """MA filter coefficients, truncated at ``m``."""
        if self.par is None:
            raise ValueError("Parameters must be set before generating MA filter")

        r, s = self.order

        if r > 0:
            psi = np.array(self.par[:r])
        else:
            psi = np.array([])

        if s > 0:
            phi = np.array(self.par[r : r + s])
        else:
            phi = np.array([])

        deltas = np.full(2 * m, np.nan)

        for k in range(-m, m):
            deltas[k + m] = madelta(psi, phi, k)

        deltas = np.flip(deltas)

        return deltas

    def _param_bounds(self, n_params: int) -> List[Tuple[float, float]]:
        """Optimisation bounds, one (lower, upper) pair per parameter."""
        # All MAR parameters are bounded between -0.99 and 0.99
        return [(-0.99, 0.99) for _ in range(n_params)]

    def generate_initial_guess(self, random: bool = False) -> List[float]:
        """Build initial parameter values; ``random`` draws them instead."""
        r, s = self.order
        p = r + s

        if random:
            ar_init = np.random.uniform(low=0.1, high=0.8, size=p)

            # rescale for stability
            if r > 0:
                psi_sum = np.sum(np.abs(ar_init[:r]))
                if psi_sum >= 0.99:
                    ar_init[:r] = ar_init[:r] * (0.9 / psi_sum)

            if s > 0:
                phi_sum = np.sum(np.abs(ar_init[r:]))
                if phi_sum >= 0.99:
                    ar_init[r:] = ar_init[r:] * (0.9 / phi_sum)
        else:
            ar_init = np.ones(p) * 0.4

            # alternate signs
            for i in range(p):
                if i % 2 == 1:
                    ar_init[i] = -ar_init[i]

        return ar_init.tolist()


# Helper functions
def madelta(
    cvec: np.ndarray, ncvec: np.ndarray, k: int, theta: float = None, eta: float = None
) -> float:
    """Infinite-MA coefficient at lag ``k``.

    ``theta``/``eta`` None or 0 means a pure MAR.
    """
    is_marma = (theta is not None and theta != 0) or (eta is not None and eta != 0)

    if not is_marma:
        if len(cvec) > 0 and not np.all(cvec == 0):
            lam = 1 / np.roots(np.flip(np.concatenate(([1], -np.array(cvec)))))
            r = len(lam)
        else:
            lam = np.zeros(0, dtype=complex)
            r = 0
        if len(ncvec) > 0 and not np.all(ncvec == 0):
            zeta = 1 / np.roots(np.flip(np.concatenate(([1], -np.array(ncvec)))))
            s = len(zeta)
        else:
            zeta = np.zeros(0, dtype=complex)
            s = 0
        delta = 0
        if k >= 0:
            for j in range(s):
                numerator = zeta[j] ** ((s - 1) + k)
                denominator1 = (
                    1
                    if s == 1
                    else np.prod([zeta[j] - zeta[i] for i in range(s) if i != j])
                )
                denominator2 = np.prod([zeta[j] * lam[i] - 1 for i in range(r)])
                if s % 2 == 0:
                    delta -= numerator / (denominator1 * denominator2)
                else:
                    delta += numerator / (denominator1 * denominator2)
        else:
            for j in range(r):
                numerator = lam[j] ** ((r - 1) - k)
                denominator1 = (
                    1
                    if r == 1
                    else np.prod([lam[j] - lam[i] for i in range(r) if i != j])
                )
                denominator2 = np.prod([lam[j] * zeta[i] - 1 for i in range(s)])
                if r % 2 == 0:
                    delta -= numerator / (denominator1 * denominator2)
                else:
                    delta += numerator / (denominator1 * denominator2)
        if (r % 2 != 0) and (s % 2 != 0):
            delta = -delta
        elif (r % 2 == 0) and (s % 2 == 0):
            delta = -delta
        return float(np.real(delta))

    else:
        # MARMA process: Fries code
        mar_coeff_k = madelta(cvec, ncvec, k, theta=None, eta=None)

        # MARMA[k] = (1+theta*eta)*MAR[k] - theta*MAR[k+1] - eta*MAR[k-1]
        mar_coeff_k_minus_1 = madelta(cvec, ncvec, k - 1, theta=None, eta=None)
        mar_coeff_k_plus_1 = madelta(cvec, ncvec, k + 1, theta=None, eta=None)

        marma_coeff = (
            (1 + theta * eta) * mar_coeff_k
            - theta * mar_coeff_k_plus_1
            - eta * mar_coeff_k_minus_1
        )

        return marma_coeff


def simulate_MARMA(
    N=1000.0,
    psi=0.9,
    phi=-0.3,
    theta=-0.4,
    eta=0.3,
    alpha=1.8,
    beta=0.5,
    sigma=0.2,
    gamma=0.0,
):
    """Simulate a MARMA process with stable innovations."""
    N = int(N)

    # Generate iid stable errors
    # scipy uses parameterization: alpha, beta, loc, scale
    eps = stable_rvs(alpha, beta, loc=gamma, scale=sigma, size=N)

    # Observations, causal/noncausal components, finite moving-average error
    y = np.zeros(N)
    u = np.zeros(N)
    v = np.zeros(N)
    z = np.zeros(N)

    z[1 : N - 1] = (
        (1 + theta * eta) * eps[1 : N - 1] - theta * eps[2:N] - eta * eps[0 : N - 2]
    )

    # Lanne-Saikkonen / Gourieroux-Jasiak decomposition builds the
    # observations from the MARMA process
    for t in range(N - 2, 1, -1):
        u[t] = psi * u[t + 1] + z[t]

        trev = N - t
        v[trev] = phi * v[trev - 1] + z[trev]

    y[1:N] = (1 / (1 - phi * psi)) * (u[1:N] + phi * v[0 : N - 1])

    trajectory = pd.Series(y, index=range(len(y)))

    return trajectory


def is_trivial_polynomial(vec, tol=1e-6):
    """Check if polynomial is trivial (constant 1): None, empty, or all zeros."""
    if vec is None:
        return True
    vec = np.atleast_1d(vec)
    if len(vec) == 0:
        return True
    if np.all(np.abs(vec) < tol):
        return True
    return False


def check_common_roots(roots1, roots2, tol=1e-6):
    """Check if two sets of roots share any common roots."""
    if len(roots1) == 0 or len(roots2) == 0:
        return False
    for r1 in roots1:
        for r2 in roots2:
            if np.abs(r1 - r2) < tol:
                return True
    return False


def check_marma_existence(psi_vec, phi_vec, theta_vec=None, eta_vec=None, tol=1e-6):
    """Check that the MARMA polynomials define an existing process."""
    # Check noncausal AR
    if is_trivial_polynomial(psi_vec, tol):
        psi_stationary = True
        psi_roots = np.array([])
    else:
        psi_vec = np.atleast_1d(psi_vec)
        psi = np.concatenate([-psi_vec[::-1], [1]])
        psi_roots = np.roots(psi)
        psi_stationary = np.all(np.abs(psi_roots) > 1.0 + tol)

    # Check causal AR
    if is_trivial_polynomial(phi_vec, tol):
        phi_stationary = True
        phi_roots = np.array([])
    else:
        phi_vec = np.atleast_1d(phi_vec)
        phi = np.concatenate([-phi_vec[::-1], [1]])
        phi_roots = np.roots(phi)
        phi_stationary = np.all(np.abs(phi_roots) > 1.0 + tol)

    # Check no common roots (noncausal)
    if is_trivial_polynomial(theta_vec, tol):
        psi_theta_common = False
    else:
        theta_vec = np.atleast_1d(theta_vec)
        theta = np.concatenate([-theta_vec[::-1], [1]])
        theta_roots = np.roots(theta)
        psi_theta_common = check_common_roots(psi_roots, theta_roots, tol)

    # Check no common roots (causal)
    if is_trivial_polynomial(eta_vec, tol):
        phi_eta_common = False
    else:
        eta_vec = np.atleast_1d(eta_vec)
        eta = np.concatenate([-eta_vec[::-1], [1]])
        eta_roots = np.roots(eta)
        phi_eta_common = check_common_roots(phi_roots, eta_roots, tol)

    exists = (
        psi_stationary
        and phi_stationary
        and not psi_theta_common
        and not phi_eta_common
    )

    return exists


def _mle_regressor_matrix(y, p):
    """Lagged regressor matrix for an AR(p) model (port of regressor.matrix)."""
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if p == 0:
        return np.zeros((n, 0))
    X = np.empty((n - p, p))
    for j in range(1, p + 1):
        X[:, j - 1] = y[p - j : n - j]
    return X


def _mle_arx_ls(y, p):
    """OLS AR(p) fit without intercept (port of arx.ls). Returns (fitted, coef)."""
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if p == 0:
        return y, np.zeros(0)
    Y = y[p:n]
    X = _mle_regressor_matrix(y, p)
    coef = np.linalg.solve(X.T @ X, X.T @ Y)
    fitted = X @ coef
    return fitted, coef


def _mle_approx(x_grid, y_grid, xout):
    """Linear interpolation with constant extrapolation (R approx rule = 2)."""
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    order = np.argsort(x_grid)
    return float(np.interp(xout, x_grid[order], y_grid[order]))


def _mle_mcculloch(x):
    """McCulloch (1986) quantile estimator (port of McCullochParametersEstim)."""
    x = np.asarray(x, dtype=float).ravel()
    q = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])

    denom_a = q[3] - q[1]
    denom_b = q[4] - q[0]
    if abs(denom_a) < 1e-10:
        denom_a = 1e-10
    if abs(denom_b) < 1e-10:
        denom_b = 1e-10

    nu_alpha = denom_b / denom_a
    nu_beta = (q[4] + q[0] - 2 * q[2]) / denom_b

    nu_alpha_grid = [
        2.439,
        2.5,
        2.6,
        2.7,
        2.8,
        3.0,
        3.2,
        3.5,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
        15.0,
        25.0,
    ]
    alpha_grid = [
        2.00,
        1.95,
        1.90,
        1.85,
        1.80,
        1.75,
        1.70,
        1.65,
        1.60,
        1.50,
        1.40,
        1.30,
        1.20,
        1.00,
        0.70,
    ]
    nu_alpha_c = max(nu_alpha_grid[0], min(nu_alpha_grid[-1], nu_alpha))
    alpha_hat = _mle_approx(nu_alpha_grid, alpha_grid, nu_alpha_c)
    alpha_hat = max(1.01, min(1.99, alpha_hat))

    alpha_c_grid = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
    c_alpha_grid = [
        0.000,
        0.010,
        0.030,
        0.060,
        0.110,
        0.170,
        0.250,
        0.340,
        0.450,
        0.600,
        0.780,
    ]
    c_alpha = _mle_approx(alpha_c_grid, c_alpha_grid, alpha_hat)
    if abs(c_alpha) < 1e-6:
        beta_hat = 0.0
    else:
        beta_hat = np.sign(nu_beta) * min(1.0, max(0.0, abs(nu_beta) / c_alpha))
    beta_hat = max(-0.999, min(0.999, beta_hat))

    alpha_s_grid = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
    psi3_grid = [
        1.908,
        1.914,
        1.921,
        1.927,
        1.933,
        1.939,
        1.946,
        1.955,
        1.965,
        1.983,
        2.000,
    ]
    psi3 = _mle_approx(alpha_s_grid, psi3_grid, alpha_hat)
    sigma_hat = max(1e-4, denom_a / psi3)

    return np.array([alpha_hat, beta_hat, sigma_hat, 0.0])


def _mle_first_pass_mcculloch(y0, p_C, p_NC):
    """McCulloch starting values on double-filtered residuals (port)."""
    y0 = np.asarray(y0, dtype=float).ravel()
    z0 = y0[::-1].copy()

    BC0 = np.array([0.0]) if p_C == 0 else np.asarray(_mle_arx_ls(y0, p_C)[1]).ravel()
    BNC0 = (
        np.array([0.0]) if p_NC == 0 else np.asarray(_mle_arx_ls(z0, p_NC)[1]).ravel()
    )

    ZC10 = y0[p_C:]
    ZC20 = _mle_regressor_matrix(y0, p_C)
    V0 = (ZC10 - ZC20 @ BC0) if p_C > 0 else ZC10

    U0 = np.asarray(V0, dtype=float)[::-1].copy()
    ZNC10 = U0[p_NC:]
    ZNC20 = _mle_regressor_matrix(U0, p_NC)
    E0 = (ZNC10 - ZNC20 @ BNC0)[::-1] if p_NC > 0 else ZNC10[::-1]

    res = _mle_mcculloch(E0 - np.mean(E0))
    res[3] = 0.0
    return res


def _mle_make_bounds(p_C, p_NC):
    """Box bounds for L-BFGS-B (port of make_bounds)."""
    n_ar = p_C + p_NC
    lower = [-0.999] * n_ar + [1.01, -0.99, 1e-4]
    upper = [0.999] * n_ar + [1.99, 0.99, 1e4]
    return lower, upper


def _mle_nll_mar_mu0(params, y, p_C, p_NC):
    """Negative log-likelihood with mu concentrated out (port of nll_mar_mu0)."""
    params = np.asarray(params, dtype=float)
    phi_c = params[:p_C] if p_C > 0 else np.zeros(0)
    phi_nc = params[p_C : p_C + p_NC] if p_NC > 0 else np.zeros(0)
    s = p_C + p_NC
    alpha = params[s]
    beta = params[s + 1]
    sigma = params[s + 2]

    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    if p_C > 0:
        ZC1 = y[p_C:n]
        ZC2 = _mle_regressor_matrix(y, p_C)
        V = ZC1 - ZC2 @ phi_c
    else:
        V = y

    if p_NC > 0:
        U = np.asarray(V, dtype=float)[::-1].copy()
        m = len(U)
        ZNC1 = U[p_NC:m]
        ZNC2 = _mle_regressor_matrix(U, p_NC)
        E = (ZNC1 - ZNC2 @ phi_nc)[::-1]
    else:
        E = np.asarray(V, dtype=float)

    mu_hat = np.mean(E)
    E_cen = E - mu_hat

    try:
        with np.errstate(all="ignore"):
            pdf_vals = stable_pdf(
                E_cen, alpha, beta, loc=0.0, scale=sigma, parametrization=0
            )
        pdf_vals[~np.isfinite(pdf_vals) | (pdf_vals <= 0)] = np.finfo(float).eps
        ll = np.sum(np.log(pdf_vals))
    except Exception:
        ll = -np.inf
    if not np.isfinite(ll):
        return 1e10
    return -ll


def _mle_init_params(y, p_C, p_NC):
    """Build the starting values in constrained space (init_params_mu0)."""
    y = np.asarray(y, dtype=float).ravel()
    z = y[::-1].copy()

    BC0 = (
        np.zeros(0)
        if p_C == 0
        else np.clip(np.asarray(_mle_arx_ls(y, p_C)[1]), 0.02, 0.93)
    )
    BNC0 = (
        np.zeros(0)
        if p_NC == 0
        else np.clip(np.asarray(_mle_arx_ls(z, p_NC)[1]), 0.02, 0.93)
    )

    mcc = _mle_first_pass_mcculloch(y, p_C, p_NC)
    alpha0 = max(1.02, min(1.98, mcc[0]))
    beta0 = max(-0.98, min(0.98, mcc[1]))
    sigma0 = max(1e-3, mcc[2])

    return np.concatenate([BC0, BNC0, [alpha0, beta0, sigma0]])


def _mle_marx_estim(y, p_C, p_NC, params0=None):
    """MAR(p_C, p_NC) MLE with mu concentrated out (port of marx.estim.alpha_sc).

    Returns a dict with keys params, mu_hat, E, Loglik (Loglik is the NLL).
    """
    y = np.asarray(y, dtype=float).ravel()
    s = p_C + p_NC

    if params0 is None:
        p0 = _mle_init_params(y, p_C, p_NC)
    else:
        params0 = np.asarray(params0, dtype=float)
        parts = []
        if p_C > 0:
            parts.append(np.clip(params0[:p_C], 0.02, 0.93))
        if p_NC > 0:
            parts.append(np.clip(params0[p_C : p_C + p_NC], 0.02, 0.93))
        parts.append(
            np.array(
                [
                    max(1.02, min(1.98, params0[s])),
                    max(-0.98, min(0.98, params0[s + 1])),
                    max(1e-3, params0[s + 2]),
                ]
            )
        )
        p0 = np.concatenate([np.atleast_1d(part) for part in parts])

    lower, upper = _mle_make_bounds(p_C, p_NC)
    bounds = list(zip(lower, upper))

    # The stable pdf (libstable via pystable, parametrization 0) matches
    # libstable4u parametrization = 0; see src.stable_mar.stable.stable_pdf.
    opt = minimize(
        _mle_nll_mar_mu0,
        p0,
        args=(y, p_C, p_NC),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e7 * np.finfo(float).eps},
    )

    params_out = opt.x
    fx = opt.fun

    # Filtered innovations and profile mu_hat at the MLE.
    n = len(y)
    phi_c_hat = params_out[:p_C] if p_C > 0 else np.zeros(0)
    phi_nc_hat = params_out[p_C : p_C + p_NC] if p_NC > 0 else np.zeros(0)

    if p_C > 0:
        ZC1 = y[p_C:n]
        ZC2 = _mle_regressor_matrix(y, p_C)
        V = ZC1 - ZC2 @ phi_c_hat
    else:
        V = y

    if p_NC > 0:
        U = np.asarray(V, dtype=float)[::-1].copy()
        m = len(U)
        ZNC1 = U[p_NC:m]
        ZNC2 = _mle_regressor_matrix(U, p_NC)
        E = (ZNC1 - ZNC2 @ phi_nc_hat)[::-1]
    else:
        E = np.asarray(V, dtype=float)

    mu_hat = np.mean(E)

    return {
        "params": params_out,
        "mu_hat": mu_hat,
        "E": np.asarray(E, dtype=float),
        "Loglik": fx,
    }


# MARMA(r, s, p, q) estimation (MLE and GCov)

MARMA_TRIM = 100


def _marma_fir(coef, x):
    """Apply (1 - sum coef[i] B^{i+1}) as an FIR filter (causal / backward)."""
    coef = np.asarray(coef, dtype=float).ravel()
    b = np.concatenate([[1.0], -coef]) if coef.size else np.array([1.0])
    return lfilter(b, [1.0], x)


def _marma_iir(coef, x):
    """Invert (1 - sum coef[i] B^{i+1}) as an IIR filter (causal / backward)."""
    coef = np.asarray(coef, dtype=float).ravel()
    a = np.concatenate([[1.0], -coef]) if coef.size else np.array([1.0])
    return lfilter([1.0], a, x)


def marma_recover_eps(y, phi_c, phi_nc, theta_c, theta_nc):
    """Recover innovations eps from y via the exact rational inverse filter.

    eps = Phi_c(B) Phi_nc(F) / [Theta_c(B) Theta_nc(F)] y. The noncausal (F)
    operators are realised by reversing the series, applying the backward filter,
    and reversing back. MA invertibility (roots of the Theta polynomials outside
    the unit circle) is required for the IIR passes to be stable.
    """
    y = np.asarray(y, dtype=float).ravel()
    a = _marma_fir(phi_c, y)  # Phi_c(B)  causal FIR
    a = _marma_fir(phi_nc, a[::-1])[::-1]  # Phi_nc(F) noncausal FIR
    h = _marma_iir(theta_c, a)  # 1/Theta_c(B)  causal IIR
    e = _marma_iir(theta_nc, h[::-1])[::-1]  # 1/Theta_nc(F) noncausal IIR
    return e


def _marma_split(params, order):
    """Split a flat dynamics vector into (phi_c, phi_nc, theta_c, theta_nc)."""
    r, s, p, q = order
    params = np.asarray(params, dtype=float).ravel()
    i = 0
    phi_c = params[i : i + r]
    i += r
    phi_nc = params[i : i + s]
    i += s
    theta_c = params[i : i + p]
    i += p
    theta_nc = params[i : i + q]
    i += q
    return phi_c, phi_nc, theta_c, theta_nc


def _marma_dynamics_bounds(order, lim=0.98):
    """Box bounds for the AR/MA coefficients (stationarity / invertibility)."""
    r, s, p, q = order
    return [(-lim, lim)] * (r + s + p + q)


def _marma_nll(params, y, order):
    """Negative alpha-stable profile log-likelihood (mu concentrated out)."""
    n_dyn = sum(order)
    phi_c, phi_nc, theta_c, theta_nc = _marma_split(params[:n_dyn], order)
    alpha, beta, sigma = params[n_dyn : n_dyn + 3]

    y = np.asarray(y, dtype=float).ravel()
    e = marma_recover_eps(y, phi_c, phi_nc, theta_c, theta_nc)
    e = e[MARMA_TRIM : len(y) - MARMA_TRIM]
    if e.size == 0 or not np.all(np.isfinite(e)):
        return 1e10
    e = e - np.mean(e)

    try:
        with np.errstate(all="ignore"):
            pdf = stable_pdf(e, alpha, beta, loc=0.0, scale=sigma, parametrization=1)
        pdf = np.asarray(pdf, dtype=float)
        pdf[~np.isfinite(pdf) | (pdf <= 0)] = np.finfo(float).eps
        ll = np.sum(np.log(pdf))
    except Exception:
        return 1e10
    return 1e10 if not np.isfinite(ll) else -ll


def _marma_gcov_criterion(eps, K, H):
    """Generalized-covariance loss sum_h Tr(Gamma_h Gamma0^-1 Gamma_h^T Gamma0^-1)."""
    eps = np.asarray(eps, dtype=float).ravel()
    E = eps[:, None] ** np.arange(1, K + 1)
    E = E - E.mean(axis=0)
    n = E.shape[0]
    G0 = (E.T @ E) / n
    try:
        iG0 = np.linalg.inv(G0)
    except np.linalg.LinAlgError:
        iG0 = np.linalg.pinv(G0)
    ls = 0.0
    for h in range(1, H + 1):
        Gh = (E[h:].T @ E[:-h]) / (n - h)
        ls += np.trace(Gh @ iG0 @ Gh.T @ iG0)
    return ls


def _marma_gcov_loss(params, y, order, K, H):
    """GCov objective over the dynamics parameters only."""
    phi_c, phi_nc, theta_c, theta_nc = _marma_split(params, order)
    y = np.asarray(y, dtype=float).ravel()
    e = marma_recover_eps(y, phi_c, phi_nc, theta_c, theta_nc)
    e = e[MARMA_TRIM : len(y) - MARMA_TRIM]
    if e.size == 0 or not np.all(np.isfinite(e)) or np.max(np.abs(e)) > 1e6 * np.std(y):
        return 1e10  # explosive / near-non-invertible region
    e = (e - e.mean()) / (np.std(e) + 1e-12)
    with np.errstate(all="ignore"):
        val = _marma_gcov_criterion(e, K, H)
    return val if np.isfinite(val) else 1e10


def _marma_default_starts(order, stable=True):
    """Deterministic start grid covering every causal/noncausal orientation.

    Same idea as the MAR grid: two neutral points, a causal-dominant and a
    noncausal-dominant AR configuration, a balanced one, and the two dominant
    orientations again with the MA coefficients pushed either way.
    """
    r, s, p, q = order
    n_ar, n_ma = r + s, p + q
    tail = [1.5, 0.0, 1.0] if stable else []
    dom, sub = 0.9, 0.1

    ar_starts = [
        [0.0] * n_ar,
        [0.2] + [0.0] * (n_ar - 1) if n_ar else [],
        [dom] * r + [sub] * s,
        [sub] * r + [dom] * s,
        [0.6] * n_ar,
    ]
    poles = [[dom] * r + [sub] * s, [sub] * r + [dom] * s]

    starts = [ar + [0.0] * n_ma + tail for ar in ar_starts]
    starts += [ar + [ma] * n_ma + tail for ma in (0.3, -0.3) for ar in poles]

    seen, unique = set(), []
    for st in starts:
        key = tuple(st)
        if key not in seen:
            seen.add(key)
            unique.append(list(st))
    return unique


def fit_marma(
    y,
    order=(1, 1, 1, 1),
    method="mle",
    start=None,
    K=2,
    H=3,
    n_restart=None,
    seed=None,
    verbose=False,
):
    """Estimate a MARMA(r, s, p, q) with alpha-stable innovations.

    ``start`` is [phi_c, phi_nc, theta_c, theta_nc, alpha, beta, sigma] for "mle"
    and the dynamics alone for "gcov"; None uses the deterministic start grid.
    """
    y = np.asarray(y, dtype=float).ravel()
    n_dyn = sum(order)

    if method == "mle":
        bounds = _marma_dynamics_bounds(order) + [
            (1.05, 1.98),
            (-0.95, 0.95),
            (1e-3, 1e3),
        ]
        starts = [start] if start is not None else _marma_default_starts(order)
        best = None
        for s0 in starts:
            try:
                opt = minimize(
                    _marma_nll,
                    s0,
                    args=(y, order),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e5 * np.finfo(float).eps},
                )
                if best is None or opt.fun < best.fun:
                    best = opt
            except Exception:
                continue
        params = best.x
        phi_c, phi_nc, theta_c, theta_nc = _marma_split(params[:n_dyn], order)
        alpha, beta, sigma = params[n_dyn : n_dyn + 3]
        e = marma_recover_eps(y, phi_c, phi_nc, theta_c, theta_nc)
        e = e[MARMA_TRIM : len(y) - MARMA_TRIM]
        out = {
            "order": order,
            "method": "mle",
            "phi_c": phi_c,
            "phi_nc": phi_nc,
            "theta_c": theta_c,
            "theta_nc": theta_nc,
            "params": params[:n_dyn],
            "alpha": float(alpha),
            "beta": float(beta),
            "sigma": float(sigma),
            "mu_hat": float(np.mean(e)),
            "E": e - np.mean(e),
            "Loglik": float(-best.fun),
        }

    elif method == "gcov":
        bounds = _marma_dynamics_bounds(order)
        starts = (
            [start] if start is not None else _marma_default_starts(order, stable=False)
        )
        best = None
        for s0 in starts:
            try:
                opt = minimize(
                    _marma_gcov_loss,
                    s0,
                    args=(y, order, K, H),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 400},
                )
                if best is None or opt.fun < best.fun:
                    best = opt
            except Exception:
                continue
        params = best.x
        phi_c, phi_nc, theta_c, theta_nc = _marma_split(params, order)
        e = marma_recover_eps(y, phi_c, phi_nc, theta_c, theta_nc)
        e = e[MARMA_TRIM : len(y) - MARMA_TRIM]
        out = {
            "order": order,
            "method": "gcov",
            "phi_c": phi_c,
            "phi_nc": phi_nc,
            "theta_c": theta_c,
            "theta_nc": theta_nc,
            "params": params,
            "mu_hat": float(np.mean(e)),
            "E": e - np.mean(e),
            "Criterion": float(best.fun),
        }
    else:
        raise ValueError("method must be 'mle' or 'gcov'")

    if verbose:
        print(f"MARMA{order} | method={method} | dynamics={np.round(out['params'], 3)}")
    return out
