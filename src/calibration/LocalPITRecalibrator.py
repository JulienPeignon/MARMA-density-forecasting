"""Local PIT recalibration of predictive densities with I-spline weights."""

import warnings

import numpy as np
import torch
import xgboost as xgb
from tqdm import tqdm

from src.calibration.ispline import get_pdf
from src.calibration.utils import cdf_from_pdf, renorm_pdf


class LocalPITRecalibrator:
    """
    Local PIT-based PDF recalibration using XGBoost and I-splines.

    Methods:
        fit(X, pit, alphas): Train the local PIT model on calibration data
        transform(X, pdf, z_grid): Apply recalibration to new PDFs
    """

    def __init__(self, n_jobs, xgb_params=None, num_basis=10):
        """Set the XGBoost parameters and the number of I-spline basis functions."""
        self.n_jobs = n_jobs
        self.xgb_params = xgb_params or self._default_xgb_params()
        self.num_basis = num_basis
        self.regressors_ = None
        self.alphas_ = None
        self.is_fitted_ = False

    def _default_xgb_params(self):
        return dict(
            max_depth=1,
            min_child_weight=100,
            learning_rate=0.03,
            n_estimators=25,
            reg_alpha=0.1,
            reg_lambda=100.0,
            gamma=0.0,
            subsample=0.8,
            colsample_bytree=1.0,
            colsample_bylevel=1.0,
            tree_method="hist",
            objective="binary:logistic",
            random_state=42,
            verbosity=0,
            n_jobs=self.n_jobs,
        )

    def fit(self, X, pit, alphas=np.linspace(0.0, 1.0, 201)):
        """Train the local PIT model on the calibration data."""
        print(f"Training local PIT model with {len(alphas)} alpha values...")
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        if torch.is_tensor(pit):
            pit = pit.detach().cpu().numpy()

        pit = np.asarray(pit).ravel()

        self.alphas_ = alphas
        self.regressors_ = []

        xgb_params = self.xgb_params.copy()

        for j, a in enumerate(tqdm(alphas, desc="Training XGB for each α")):
            y_train_bin = (pit <= a).astype(int)

            if np.all(y_train_bin == 0):
                self.regressors_.append(lambda x: np.zeros(len(x)))
            elif np.all(y_train_bin == 1):
                self.regressors_.append(lambda x: np.ones(len(x)))
            else:
                clf = xgb.XGBClassifier(**xgb_params)

                clf.fit(X, y_train_bin)

                self.regressors_.append(clf)

        self.is_fitted_ = True
        return self

    def _predict_local_pit(self, X):
        """Predict local PIT distribution for new data."""
        if not self.is_fitted_:
            raise RuntimeError(
                "Model must be fitted before transform. Call .fit() first."
            )

        beta = np.zeros((len(X), len(self.alphas_)), dtype=float)

        for j, clf in enumerate(self.regressors_):
            if callable(clf) and not hasattr(clf, "predict_proba"):
                beta[:, j] = clf(X)
            else:
                beta[:, j] = clf.predict_proba(X)[:, 1]

        beta = np.clip(beta, 0.0, 1.0)
        return beta

    def _get_local_correction(self, beta_alpha, F_grid):
        """Compute M-spline derivative correction factors."""
        N, K = F_grid.shape
        corr = np.zeros((N, K), dtype=float)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message=".*encountered in matmul"
            )

            residuals = np.empty(N, dtype=float)
            for i in range(N):
                corr[i], _, residuals[i] = get_pdf(
                    self.alphas_, beta_alpha[i], F_grid[i], num_basis=self.num_basis
                )

        n_invalid = np.sum(~np.isfinite(corr))
        if n_invalid > 0:
            raise RuntimeError(
                f"M-spline PDF computation failed: {n_invalid}/{corr.size} "
                f"invalid values. Try reducing num_basis "
                f"(current: {self.num_basis})."
            )

        self.fit_residuals_ = residuals
        if residuals.mean() > 0.15:
            warnings.warn(
                f"local PIT map poorly approximated: mean |spline - beta| = "
                f"{residuals.mean():.3f} (max {residuals.max():.3f}) at "
                f"num_basis={self.num_basis}; raise num_basis.",
                RuntimeWarning,
                stacklevel=2,
            )

        return np.clip(corr, 0.0, None)

    def transform(self, X, pdf, z_grid, verbose=True):
        """Recalibrate ``pdf`` on ``z_grid`` for the features ``X``."""
        if not self.is_fitted_:
            raise RuntimeError(
                "Model must be fitted before transform. Call .fit() first."
            )

        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        if torch.is_tensor(pdf):
            pdf = pdf.detach().cpu().numpy()
        else:
            pdf = np.asarray(pdf)

        if verbose:
            print(f"Recalibrating {len(X)} PDFs...")

        F_hat = cdf_from_pdf(pdf, z_grid)

        beta_alpha = self._predict_local_pit(X)

        correction = self._get_local_correction(beta_alpha, F_hat)

        pdf_corrected = np.multiply(pdf, correction)
        pdf_recal = renorm_pdf(pdf_corrected, z_grid)

        return pdf_recal
