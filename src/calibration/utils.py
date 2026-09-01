"""CDF and PIT helpers shared by the recalibration code."""

import numpy as np
import torch
from scipy.integrate import trapezoid


def cdf_from_pdf(pdf, z):
    """CDF on the same z-grid; guards overshoot to [0,1]."""
    if torch.is_tensor(pdf):
        pdf = pdf.detach().cpu().numpy()
    else:
        pdf = np.asarray(pdf)

    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()
    else:
        z = np.asarray(z)

    dz = np.gradient(np.squeeze(z))
    cdf = np.cumsum(pdf * dz, axis=-1)
    return np.clip(cdf, 0.0, 1.0 - 1e-12)


def renorm_pdf(pdf, z, eps=1e-12):
    """Renormalize PDF to unit area."""
    pdf = np.asarray(pdf)
    z = np.asarray(z)

    area = trapezoid(pdf, z, axis=-1)
    area = np.maximum(area, eps)
    area = area[..., None]
    return pdf / area
