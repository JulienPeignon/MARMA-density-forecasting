"""Mixture density network with Azzalini skew-Student-t components."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import Parallel, delayed
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_is_fitted
from torch.special import gammaln
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

from src.utils.setup_logger import setup_logger

NU_MIN = 2.0
NU_MAX = 100.0
NU_INIT = 5.0
G_MAX = 6.0


def check_tensor(x, dtype=torch.float32):
    """Return ``x`` as a tensor of ``dtype``, without re-wrapping if it is one."""
    if isinstance(x, torch.Tensor):
        return x if x.dtype == dtype else x.to(dtype=dtype)
    return torch.tensor(x, dtype=dtype)


def betaln(a, b):
    """Log of beta function."""
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)


def betainc(a, b, x):
    """Regularised incomplete beta function in PyTorch."""
    device = x.device
    dtype = x.dtype

    eps = torch.finfo(dtype).eps
    EPSILON = torch.tensor(eps * 1e3, dtype=dtype, device=device)
    MAX_ITER = 20  # fixed loop

    nan_mask = (a < 0.0) | (b < 0.0)
    zero_mask = x < 0.0
    one_mask = x > 1.0

    inversion_mask = x > (a + 1.0) / (a + b + 2.0)
    a, b = (
        torch.where(inversion_mask, b, a),
        torch.where(inversion_mask, a, b),
    )
    x = torch.where(inversion_mask, 1.0 - x, x)

    front = torch.exp(torch.log(x) * a + torch.log(1.0 - x) * b - betaln(a, b)) / a

    f = torch.ones_like(a)
    c = torch.ones_like(a)
    d = torch.zeros_like(a)

    for i in range(MAX_ITER):
        m = i // 2

        if i == 0:
            numerator = torch.ones_like(a)
        elif i % 2 == 0:  # Even iterations
            numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:  # Odd iterations
            numerator = -((a + m) * (a + b + m) * x) / (
                (a + 2.0 * m) * (a + 2.0 * m + 1)
            )

        d = 1.0 + numerator * d
        d = torch.where(torch.abs(d) < EPSILON, EPSILON, d)
        d = 1.0 / d

        c = 1.0 + numerator / c
        c = torch.where(torch.abs(c) < EPSILON, EPSILON, c)

        f = f * (c * d)

    result = torch.where(inversion_mask, 1 - front * (f - 1.0), front * (f - 1.0))

    result = torch.where(nan_mask, torch.full_like(result, float("nan")), result)
    result = torch.where(zero_mask, torch.zeros_like(result), result)
    result = torch.where(one_mask, torch.ones_like(result), result)

    return result


def student_cdf(x, nu):
    """Student-t CDF, through the regularised incomplete beta."""
    x = check_tensor(x)
    nu = check_tensor(nu)
    nu = nu.to(x.device)

    nu_expanded = nu.expand_as(x)
    eps = torch.finfo(x.dtype).eps
    tiny = torch.finfo(x.dtype).tiny
    z = (nu_expanded / (nu_expanded + x * x)).clamp(min=tiny, max=1.0 - eps)

    # 1/2 * I_z(nu/2, 1/2)
    half_I = 0.5 * betainc(nu_expanded / 2, torch.full_like(x, 0.5), z)

    return torch.where(x <= 0, half_I, 1.0 - half_I)


def skewt_pdf(y_grid, mu, sigma, nu, skew):
    """Azzalini (2014) skew-Student-t density.

    f(x) = 2 f_t(x|nu) F_t(alpha x sqrt((nu+1)/(nu+x^2)) | nu+1)
    """
    y_grid = check_tensor(y_grid)
    mu = check_tensor(mu)
    sigma = check_tensor(sigma)
    nu = check_tensor(nu)
    skew = check_tensor(skew)

    device = y_grid.device
    mu = mu.to(device)
    sigma = sigma.to(device)
    nu = nu.to(device)
    skew = skew.to(device)

    z = (y_grid - mu) / sigma

    log_norm = (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * torch.log(nu * torch.tensor(torch.pi, device=device))
    )
    log_kernel = -(nu + 1) / 2 * torch.log(1 + z**2 / nu)
    ft_z = torch.exp(log_norm + log_kernel)

    # Argument for the CDF: alpha * z * sqrt((nu+1)/(nu+z^2))
    cdf_arg = skew * z * torch.sqrt((nu + 1) / (nu + z**2))

    Ft_arg = student_cdf(cdf_arg, nu + 1)

    # Skew-t PDF: 2 * f_t(z|nu) * F_t(cdf_arg|nu+1) / sigma
    pdf = 2 * ft_z * Ft_arg / sigma

    return pdf


def _log_student_pdf_std(x, nu):
    """Log of the standard Student-t pdf f_t(x; nu) (location 0, scale 1)."""
    x = check_tensor(x)
    nu = check_tensor(nu).to(x.device)
    pi_t = torch.tensor(np.pi, dtype=x.dtype, device=x.device)
    return (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * torch.log(nu * pi_t)
        - (nu + 1) / 2 * torch.log1p(x * x / nu)
    )


def _log_student_cdf_value(x, nu):
    """Log F_t(x; nu) value, stable in the lower tail (no analytic x-gradient)."""
    x = check_tensor(x)
    nu = check_tensor(nu).to(x.device)
    nu = nu.expand_as(x) if nu.dim() else nu

    tiny = torch.finfo(x.dtype).tiny
    Ft = student_cdf(x, nu)
    direct = torch.log(Ft.clamp_min(tiny))

    # Asymptotic lower-tail branch
    pi_t = torch.tensor(np.pi, dtype=x.dtype, device=x.device)
    ax = x.abs().clamp_min(tiny)
    log_K = (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * torch.log(nu * pi_t)
        + (nu + 1) / 2 * torch.log(nu)
    )
    asymp = log_K - torch.log(nu) - nu * torch.log(ax)

    use_asymp = (x < 0) & (Ft <= tiny)
    return torch.where(use_asymp, asymp, direct)


def log_student_cdf(x, nu):
    """Log Student-t CDF; stable and correctly differentiable."""
    x = check_tensor(x)
    nu = check_tensor(nu).to(x.device)

    # Value + correct nu-gradient; x detached so its unreliable autograd x-gradient
    # is discarded (the straight-through term below supplies the x-gradient).
    log_F_val = _log_student_cdf_value(x.detach(), nu)

    # Exact x-gradient = hazard score f_t/F_t, as a detached constant. Computed in
    # log-space so it stays finite where f_t and F_t both underflow (tail).
    log_ft = _log_student_pdf_std(x, nu)
    log_F_det = _log_student_cdf_value(x, nu).detach()
    score = torch.exp(log_ft.detach() - log_F_det)

    # Straight-through: value unchanged (x - x.detach() == 0), x-gradient == score.
    return log_F_val + score * (x - x.detach())


def mixture_nll(log_pi, mu, sigma, nu, skew, target, weights):
    """Negative log-likelihood of an Azzalini skew-Student-t mixture.

    All parameter tensors are (batch_size, n_mixtures); ``weights`` reweights the
    per-observation terms.
    """
    batch_size, n_mixtures = mu.shape

    target_expanded = target.unsqueeze(1).expand(batch_size, n_mixtures)

    # Standardize: z_tilde = (Y - mu) / sigma
    z_tilde = (target_expanded - mu) / sigma

    log_sigma = torch.log(sigma)
    log_normalization = (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5
        * torch.log(nu * torch.tensor(np.pi, dtype=torch.float32, device=nu.device))
    )
    log_kernel = -(nu + 1) / 2 * torch.log(1 + z_tilde**2 / nu)
    log_ft = log_normalization + log_kernel  # log f_t(z_tilde | nu)

    # CDF argument: lambda * z_tilde * sqrt((nu+1)/(nu+z_tilde^2))
    cdf_arg = skew * z_tilde * torch.sqrt((nu + 1) / (nu + z_tilde**2))

    # log F_t(cdf_arg | nu+1) with a stable lower-tail asymptotic (no clamp),
    # so the skewness gradient survives in the left tail.
    log_Ft = log_student_cdf(cdf_arg, nu + 1)

    # log pdf = log(2) + log(f_t) + log(F_t) - log(sigma)
    log_prob_components = np.log(2) + log_ft + log_Ft - log_sigma

    # Add log mixture weights (exact; no additive epsilon)
    weighted_log_prob = log_pi + log_prob_components

    log_sum_exp = torch.logsumexp(weighted_log_prob, dim=1)

    loss = (-log_sum_exp * weights).mean()

    return loss


class MixtureDensityNetwork(nn.Module):
    """MDN with Azzalini skew-Student-t components for heavy-tailed targets."""

    def __init__(
        self,
        input_dim,
        hidden_layers,
        n_mixtures,
        device="cpu",
        n_jobs=1,
        dropout=0.0,
    ):
        """Build the network."""
        super().__init__()
        self.input_dim = input_dim
        self.n_mixtures = n_mixtures
        self.device = torch.device(device)
        self.n_jobs = n_jobs
        self.dropout = dropout

        self.scaler_x = RobustScaler()
        self.scaler_y = RobustScaler()

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            linear = nn.Linear(in_dim, hidden_dim)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)

        self.pi_layer = nn.Linear(hidden_layers[-1], n_mixtures)

        self.mu_layer = nn.Linear(hidden_layers[-1], n_mixtures)
        self.sigma_layer = nn.Linear(hidden_layers[-1], n_mixtures)

        self.nu_layer = nn.Linear(hidden_layers[-1], n_mixtures)
        self.lam_layer = nn.Linear(hidden_layers[-1], n_mixtures)

        nn.init.zeros_(self.nu_layer.weight)
        nn.init.constant_(
            self.nu_layer.bias,
            float(np.log((NU_INIT - NU_MIN) / (NU_MAX - NU_INIT))),
        )
        nn.init.zeros_(self.lam_layer.weight)
        nn.init.zeros_(self.lam_layer.bias)

    def forward(self, x):
        """Return the mixture parameters (pi, log_pi, mu, sigma, nu, skew)."""
        x = x.to(self.device)
        hidden_state = self.hidden(x)

        log_pi = F.log_softmax(self.pi_layer(hidden_state), dim=-1)
        pi = log_pi.exp()

        mu = self.mu_layer(hidden_state)
        sigma = F.softplus(self.sigma_layer(hidden_state)) + 1e-6  # Ensure positivity

        nu = NU_MIN + (NU_MAX - NU_MIN) * torch.sigmoid(self.nu_layer(hidden_state))
        skew = G_MAX * torch.tanh(self.lam_layer(hidden_state))

        return pi, log_pi, mu, sigma, nu, skew

    def normalize_data(self, X, y):
        """Fit the scalers on the training data and return the scaled tensors."""
        if X.dim() == 1:
            X = X.unsqueeze(1)
        if y.dim() > 1:
            y = y.squeeze()

        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy().reshape(-1, 1)

        # y is always globally scaled
        self.scaler_y.fit(y_np)
        y_norm = torch.tensor(
            self.scaler_y.transform(y_np).ravel(), dtype=torch.float32, device=y.device
        )

        self.scaler_x.fit(X_np)
        X_norm = torch.tensor(
            self.scaler_x.transform(X_np), dtype=torch.float32, device=X.device
        )

        return X_norm, y_norm

    def renormalize_test(self, X, y=None):
        """Scale test data with the fitted training scalers."""
        if X.dim() == 1:
            X = X.unsqueeze(1)

        check_is_fitted(self.scaler_x)
        X_norm = torch.tensor(
            self.scaler_x.transform(X.detach().cpu().numpy()),
            dtype=torch.float32,
            device=X.device,
        )

        if y is not None:
            if y.dim() > 1:
                y = y.squeeze()
            y_norm = torch.tensor(
                self.scaler_y.transform(
                    y.detach().cpu().numpy().reshape(-1, 1)
                ).ravel(),
                dtype=torch.float32,
                device=y.device,
            )
            return X_norm, y_norm
        return X_norm

    def denormalize_params(self, pi, mu, sigma, nu, skew):
        """Map the mixture parameters back to the original scale."""
        mu_y = torch.tensor(self.scaler_y.center_[0], dtype=mu.dtype, device=mu.device)
        sigma_y = torch.tensor(
            self.scaler_y.scale_[0], dtype=sigma.dtype, device=sigma.device
        )

        mu_denorm = mu_y + sigma_y * mu
        sigma_denorm = sigma_y * sigma

        return pi, mu_denorm, sigma_denorm, nu, skew

    def prepare_dataloaders(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        weights_train=None,
        weights_val=None,
        sampler=None,
        batch_size=256,
    ):
        """Build the training and validation loaders."""
        train_dataset = TensorDataset(
            X_train.detach().cpu(), y_train.detach().cpu(), weights_train.detach().cpu()
        )
        test_dataset = TensorDataset(
            X_val.detach().cpu(), y_val.detach().cpu(), weights_val.detach().cpu()
        )

        if sampler:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
            )

        val_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, val_loader

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        weights_train,
        weights_val,
        max_epochs=200,
        learning_rate=1e-3,
        batch_size=256,
        max_norm=50.0,
        patience=20,
        scheduler_patience=10,
        scheduler_factor=0.5,
    ):
        """Fit the network on normalised data."""
        logger = setup_logger()

        logger.info("Normalizing data...")
        X_train_norm, y_train_norm = self.normalize_data(X_train, y_train)

        X_val_norm, y_val_norm = self.renormalize_test(X_val, y_val)

        logger.info(
            f"X: center={self.scaler_x.center_.mean():.3f}, "
            f"scale={self.scaler_x.scale_.mean():.3f}"
        )
        logger.info(
            f"y: center={self.scaler_y.center_[0]:.3f}, "
            f"scale={self.scaler_y.scale_[0]:.3f}"
        )

        sampler_training = WeightedRandomSampler(
            weights_train, num_samples=len(X_train_norm), replacement=True
        )

        w_train = weights_train.detach().cpu().float()
        loss_w_divisor = float((w_train * w_train).mean() / w_train.mean())
        w_val = weights_val.detach().cpu().float()
        val_w_divisor = float((w_val * w_val).mean())
        logger.info(
            f"Weight scaling: train divisor={loss_w_divisor:.3f}, "
            f"val divisor={val_w_divisor:.3f}"
        )

        train_loader, val_loader = self.prepare_dataloaders(
            X_train=X_train_norm,
            y_train=y_train_norm,
            X_val=X_val_norm,
            y_val=y_val_norm,
            weights_train=weights_train,
            weights_val=weights_val,
            sampler=sampler_training,
            batch_size=batch_size,
        )

        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.RAdam(
            trainable,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-6,
            weight_decay=1e-4,
            decoupled_weight_decay=True,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
        )

        train_losses, val_losses, val_losses_unweighted = [], [], []

        grad_norms = []
        clip_events = 0
        total_steps = 0
        skipped_batches = 0

        best_val_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        percent_trainable = 100 * trainable_params / total_params
        logger.info(
            f"Trainable parameters: {trainable_params}/{total_params} "
            f"({percent_trainable:.1f}%)"
        )

        for epoch in range(max_epochs):
            self.train()

            train_loss = 0.0
            for X_batch, y_batch, weights_batch in train_loader:
                total_steps += 1

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                weights_batch = weights_batch.to(self.device)

                optimizer.zero_grad()

                pi, log_pi, mu, sigma, nu, skew = self(X_batch)

                loss_weights = weights_batch / loss_w_divisor
                loss = mixture_nll(log_pi, mu, sigma, nu, skew, y_batch, loss_weights)

                if not torch.isfinite(loss):
                    skipped_batches += 1
                    logger.warning(f"Skipping batch {total_steps}: non-finite loss")
                    continue

                loss.backward()

                clipped_norm = nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=max_norm
                )

                if not torch.isfinite(clipped_norm):
                    skipped_batches += 1
                    logger.warning(
                        f"Skipping batch {total_steps}: non-finite gradients"
                    )
                    optimizer.zero_grad()
                    continue

                grad_norms.append(float(clipped_norm))
                if clipped_norm > max_norm:
                    clip_events += 1

                optimizer.step()
                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)

            self.eval()
            val_loss_w = 0.0
            val_loss_unw = 0.0

            with torch.no_grad():
                for X_val_b, y_val_b, weights_val_b in val_loader:
                    X_val_b = X_val_b.to(self.device)
                    y_val_b = y_val_b.to(self.device)
                    weights_val_b = weights_val_b.to(self.device)

                    pi, log_pi, mu, sigma, nu, skew = self(X_val_b)

                    # Weighted val loss, on the same w^2 tilt as the training loss
                    val_weights = weights_val_b**2 / val_w_divisor
                    loss_w = mixture_nll(
                        log_pi, mu, sigma, nu, skew, y_val_b, val_weights
                    )
                    val_loss_w += loss_w.item() * len(X_val_b)

                    ones = torch.ones(len(X_val_b), device=self.device)
                    loss_unw = mixture_nll(log_pi, mu, sigma, nu, skew, y_val_b, ones)
                    val_loss_unw += loss_unw.item() * len(X_val_b)

            val_loss_w /= len(val_loader.dataset)
            val_loss_unw /= len(val_loader.dataset)

            val_losses.append(val_loss_w)
            val_losses_unweighted.append(val_loss_unw)

            # Early stopping on weighted validation loss
            early_stop_loss = val_loss_w

            scheduler.step(early_stop_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch + 1:02d} | LR: {current_lr:.1e} | "
                f"Train Loss: {train_loss:.3f} | "
                f"Val Loss: {val_loss_w:.3f} ({val_loss_unw:.3f})"
            )

            if early_stop_loss < best_val_loss:
                best_val_loss = early_stop_loss
                best_model_state = copy.deepcopy(self.state_dict())
                epochs_without_improvement = 0
                logger.info(f"- New best model saved (val_loss: {best_val_loss:.3f}) -")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                    logger.info(
                        f"Restored best model with val_loss: {best_val_loss:.3f}"
                    )
                break

        grad_norms_tensor = torch.tensor(grad_norms)
        logger.info(f"\n{'=' * 30}")
        logger.info("GRADIENT NORM DIAGNOSIS")
        logger.info(f"{'=' * 30}")
        logger.info(f"Total gradient updates: {total_steps}")
        logger.info(
            f"Skipped batches: {skipped_batches}/{total_steps} "
            f"({100 * skipped_batches / total_steps:.1f}%)"
        )
        logger.info(f"Average gradient norm: {grad_norms_tensor.mean():.3f}")
        logger.info(f"Median gradient norm: {grad_norms_tensor.median():.3f}")
        logger.info(f"Max gradient norm: {grad_norms_tensor.max():.3f}")
        logger.info(
            f"Gradient clipping events: {clip_events}/{total_steps} "
            f"({100 * clip_events / total_steps:.1f}%)"
        )
        logger.info(f"Clipping threshold: {max_norm}")

        denom = max(total_steps, 1)
        self.last_fit_diagnostics = {
            "total_steps": int(total_steps),
            "clip_events": int(clip_events),
            "skipped_batches": int(skipped_batches),
            "pct_clipped": 100.0 * clip_events / denom,
            "pct_skipped": 100.0 * skipped_batches / denom,
        }

        return best_val_loss

    def pred(self, X, grid):
        """Predict densities on ``grid``, on the original scale."""
        device = self.device
        dtype = torch.float32
        self.eval()

        with torch.no_grad():
            X = X.to(device)
            check_is_fitted(self.scaler_x)
            X_norm = torch.tensor(
                self.scaler_x.transform(X.detach().cpu().numpy()),
                dtype=torch.float32,
                device=device,
            )

            pi, _, mu_norm, sigma_norm, nu, skew = self.forward(X_norm)

            check_is_fitted(self.scaler_y)
            pi, mu, sigma, nu, skew = self.denormalize_params(
                pi, mu_norm, sigma_norm, nu, skew
            )

            batch_size = X.shape[0]

            pi_cpu = pi.detach().cpu()
            mu_cpu = mu.detach().cpu()
            sigma_cpu = sigma.detach().cpu()
            nu_cpu = nu.detach().cpu()
            lam_cpu = skew.detach().cpu()

            grid_t = torch.as_tensor(grid, dtype=dtype)
            density = torch.zeros(batch_size, grid_t.numel(), dtype=dtype)

            def _one_row(i):
                row = torch.zeros(grid_t.numel(), dtype=dtype)
                for k in range(self.n_mixtures):
                    comp_pdf = skewt_pdf(
                        grid_t,
                        mu_cpu[i, k],
                        sigma_cpu[i, k],
                        nu_cpu[i, k],
                        lam_cpu[i, k],
                    )
                    row += pi_cpu[i, k] * comp_pdf
                return i, row

            if self.n_jobs == 1:
                for i in tqdm(range(batch_size), desc="Computing density", leave=False):
                    _, row = _one_row(i)
                    density[i] = row
            else:
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(_one_row)(i)
                    for i in tqdm(
                        range(batch_size), desc="Computing density", leave=False
                    )
                )
                for i, row in results:
                    density[i] = row

            return density.to(device)
