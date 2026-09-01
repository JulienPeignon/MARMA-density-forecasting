"""Conditional moments of a stable MAR/MARMA process, by Fourier inversion."""

import warnings

import numpy as np
from scipy import integrate

from src.stable_mar.stable import stable_pdf
from src.stable_mar.stable_mar import madelta


def compute_theoretical_moments(
    grid,
    h=1,
    phi_vec=[0.3],
    psi_vec=[0.9],
    alpha=1.8,
    beta=0.0,
    sigma=0.2,
    theta=None,
    eta=None,
    ma_trunc=100,
):
    """Conditional moments up to fourth order, evaluated on ``grid``."""
    # Determine maximum order to compute (strict inequality: order < 2*alpha + 1)
    max_order = min(4, int(np.ceil(2 * alpha + 1)) - 1)

    coeff_ma = np.array(
        [
            madelta(cvec=phi_vec, ncvec=psi_vec, k=k, theta=theta, eta=eta)
            for k in range(-ma_trunc, ma_trunc)
        ],
        dtype=float,
    )

    sig1 = (sigma**alpha) * np.sum(np.abs(coeff_ma) ** alpha)
    bet1 = (
        beta
        * np.sum(np.sign(coeff_ma) * np.abs(coeff_ma) ** alpha)
        / np.sum(np.abs(coeff_ma) ** alpha)
    )

    tg = np.tan(np.pi * alpha / 2)

    varxlow = grid
    varxupp = np.array([])

    remember_esp = np.zeros(len(grid))
    remember_sec = np.zeros(len(grid)) if max_order >= 2 else None
    remember_thir = np.zeros(len(grid)) if max_order >= 3 else None
    remember_four = np.zeros(len(grid)) if max_order >= 4 else None

    H1 = np.zeros(len(varxlow))
    Hx2 = np.zeros((len(varxlow), 2)) if max_order >= 2 else None
    Hx3 = np.zeros((len(varxlow), 2)) if max_order >= 3 else None
    Hx4 = np.zeros((len(varxlow), 2)) if max_order >= 4 else None

    fX = stable_pdf(varxlow, alpha, bet1, 0, sig1 ** (1 / alpha))

    def integrate_func(vart, x, func_type, power=0, trig="sin"):
        """Integrand of the inversion formula for the requested moment."""
        exp_part = np.exp(-sig1 * vart**alpha)
        phase = vart * x - tg * bet1 * sig1 * vart**alpha

        if func_type == "H1":
            return exp_part * np.sin(phase)
        elif func_type == "Hx":
            if trig == "cos":
                return exp_part * np.cos(phase) * vart ** (power * (alpha - 1))
            else:  # sin
                return exp_part * np.sin(phase) * vart ** (power * (alpha - 1))

    for i in range(len(varxlow)):
        vart = np.arange(0, 15.01, 0.01)
        valyH = integrate_func(vart, varxlow[i], "H1")
        H1[i] = integrate.simpson(valyH, vart)

    # Compute Hx2 functions (only if order >= 2)
    if max_order >= 2:
        for i in range(len(varxlow)):
            vart = np.arange(0, 15.01, 0.01)

            # cosine
            valyH = integrate_func(vart, varxlow[i], "Hx", power=2, trig="cos")
            Hx2[i, 0] = integrate.simpson(valyH, vart)

            # sine
            valyH = integrate_func(vart, varxlow[i], "Hx", power=2, trig="sin")
            Hx2[i, 1] = integrate.simpson(valyH, vart)

    # Compute Hx3 functions (only if order >= 3)
    if max_order >= 3:
        for i in range(len(varxlow)):
            vart = np.arange(0, 15.01, 0.01)

            # cosine
            valyH = integrate_func(vart, varxlow[i], "Hx", power=3, trig="cos")
            Hx3[i, 0] = integrate.simpson(valyH, vart)

            # sine
            valyH = integrate_func(vart, varxlow[i], "Hx", power=3, trig="sin")
            Hx3[i, 1] = integrate.simpson(valyH, vart)

    # Compute Hx4 functions (only if order >= 4)
    if max_order >= 4:
        for i in range(len(varxlow)):
            vart = np.arange(0, 15.01, 0.01)

            # cosine
            valyH = integrate_func(vart, varxlow[i], "Hx", power=4, trig="cos")
            Hx4[i, 0] = integrate.simpson(valyH, vart)

            # sine
            valyH = integrate_func(vart, varxlow[i], "Hx", power=4, trig="sin")
            Hx4[i, 1] = integrate.simpson(valyH, vart)

    idxlow = np.arange(len(varxlow))
    idxupp = np.array([], dtype=int)

    coeff_ma_0 = coeff_ma[h : len(coeff_ma)]
    coeff_ma_h = coeff_ma[0 : len(coeff_ma) - h]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        k1 = np.nansum(
            np.abs(coeff_ma_0) ** alpha * (coeff_ma_h / coeff_ma_0)
        ) / np.sum(np.abs(coeff_ma) ** alpha)
        k2 = (
            np.nansum(np.abs(coeff_ma_0) ** alpha * (coeff_ma_h / coeff_ma_0) ** 2)
            / np.sum(np.abs(coeff_ma) ** alpha)
            if max_order >= 2
            else None
        )
        k3 = (
            np.nansum(np.abs(coeff_ma_0) ** alpha * (coeff_ma_h / coeff_ma_0) ** 3)
            / np.sum(np.abs(coeff_ma) ** alpha)
            if max_order >= 3
            else None
        )
        k4 = (
            np.nansum(np.abs(coeff_ma_0) ** alpha * (coeff_ma_h / coeff_ma_0) ** 4)
            / np.sum(np.abs(coeff_ma) ** alpha)
            if max_order >= 4
            else None
        )

        l1 = (
            beta
            * np.nansum(
                np.sign(coeff_ma_0)
                * np.abs(coeff_ma_0) ** alpha
                * (coeff_ma_h / coeff_ma_0)
            )
            / np.sum(np.abs(coeff_ma) ** alpha)
        )
        l2 = (
            (
                beta
                * np.nansum(
                    np.sign(coeff_ma_0)
                    * np.abs(coeff_ma_0) ** alpha
                    * (coeff_ma_h / coeff_ma_0) ** 2
                )
                / np.sum(np.abs(coeff_ma) ** alpha)
            )
            if max_order >= 2
            else None
        )
        l3 = (
            (
                beta
                * np.nansum(
                    np.sign(coeff_ma_0)
                    * np.abs(coeff_ma_0) ** alpha
                    * (coeff_ma_h / coeff_ma_0) ** 3
                )
                / np.sum(np.abs(coeff_ma) ** alpha)
            )
            if max_order >= 3
            else None
        )
        l4 = (
            (
                beta
                * np.nansum(
                    np.sign(coeff_ma_0)
                    * np.abs(coeff_ma_0) ** alpha
                    * (coeff_ma_h / coeff_ma_0) ** 4
                )
                / np.sum(np.abs(coeff_ma) ** alpha)
            )
            if max_order >= 4
            else None
        )

    kl = k1**2 - tg**2 * l1**2

    remember_esp[idxlow] = varxlow * k1 + tg * (
        (l1 - bet1 * k1) / (1 + tg**2 * bet1**2)
    ) * (tg * bet1 * varxlow + (1 - varxlow * H1) / (np.pi * fX))

    if len(idxupp) > 0:
        if np.abs(bet1) < 1:
            remember_esp[idxupp] = (
                varxupp * (k1 + np.sign(varxupp) * l1) / (1 + np.sign(varxupp) * bet1)
            )
        elif np.abs(bet1) == 1:
            remember_esp[idxupp] = varxupp * k1
        else:
            raise ValueError("Beta_1 not within [-1,1]")

    if max_order >= 2:
        nuvar21 = k2 - tg**2 * bet1 * l2 - k1**2 + tg**2 * l1**2
        nuvar22 = 2 * tg * l1 * k1 - tg * (l2 + bet1 * k2)

        remember_sec[idxlow] = (varxlow / (1 + tg**2 * bet1**2)) * (
            varxlow * (tg**2 * l2 * bet1 + k2)
            + tg * (l2 - k2 * bet1) * (1 - varxlow * H1) / (np.pi * fX)
        ) + (alpha**2 * sig1**2 / (np.pi * fX)) * (
            nuvar21 * Hx2[:, 0] + nuvar22 * Hx2[:, 1]
        )

        if len(idxupp) > 0:
            if np.abs(bet1) < 1:
                remember_sec[idxupp] = (
                    varxupp**2
                    * (k2 + np.sign(varxupp) * l2)
                    / (1 + np.sign(varxupp) * bet1)
                )
            elif np.abs(bet1) == 1:
                remember_sec[idxupp] = varxupp**2 * k2
            else:
                raise ValueError("Beta_1 not within [-1,1]")

    if max_order >= 3:
        K = k1 * l2 + k2 * l1
        L = k1 * k2 - tg**2 * l1 * l2

        # nu_I's
        nu11 = k3
        nu12 = -tg * l3
        nu21 = L
        nu22 = -tg * K
        nu31 = tg * l1 * (3 * k1**2 - tg**2 * l1**2)
        nu32 = k1**3 - 3 * tg**2 * k1 * l1**2
        nu41 = tg * (bet1 * L + K)
        nu42 = L - tg**2 * bet1 * K
        nu51 = tg * K
        nu52 = L
        nu61 = tg * (l3 + bet1 * k3)
        nu62 = k3 - tg**2 * bet1 * l3

        # nu_K's
        nuK11 = nu11
        nuK12 = nu12
        nuK21 = nu21
        nuK22 = nu22
        nuK31 = nu31 - nu41
        nuK32 = nu32 - nu42
        nuK41 = nu61 - nu51
        nuK42 = nu62 - nu52

        # Final nu's
        nuske21 = -2 * (nuK11 + tg * bet1 * nuK12) + 2 * nuK21 - nuK42
        nuske22 = -2 * (nuK12 - tg * bet1 * nuK11) + 2 * nuK22 + nuK41
        nuske31 = 2 * nuK31 + nuK41 + tg * bet1 * nuK42
        nuske32 = 2 * nuK32 + nuK42 - tg * bet1 * nuK41

        remember_thir[idxlow] = (varxlow**2 / (1 + tg**2 * bet1**2)) * (
            varxlow * (nuK11 - tg * bet1 * nuK12)
            - (1 - varxlow * H1) * (nuK12 + tg * bet1 * nuK11) / (np.pi * fX)
        ) - (alpha / (np.pi * fX)) * (
            alpha * varxlow * sig1**2 * (nuske21 * Hx2[:, 0] + nuske22 * Hx2[:, 1]) / 2
            + alpha**2 * sig1**3 * (nuske31 * Hx3[:, 0] + nuske32 * Hx3[:, 1]) / 2
        )

        if len(idxupp) > 0:
            if np.abs(bet1) < 1:
                remember_thir[idxupp] = (
                    varxupp**3
                    * (k3 + np.sign(varxupp) * l3)
                    / (1 + np.sign(varxupp) * bet1)
                )
            elif np.abs(bet1) == 1:
                remember_thir[idxupp] = varxupp**3 * k3
            else:
                raise ValueError("Beta_1 not within [-1,1]")

    if max_order >= 4:
        K = k1 * l3 + k3 * l1
        L = k1 * k3 - tg**2 * l1 * l3

        # nu of the J's
        nu11 = tg * (l2 * kl + 2 * k1 * k2 * l1)
        nu12 = k2 * kl - 2 * tg**2 * k1 * l1 * l2
        nu21 = tg * (K + bet1 * L)
        nu22 = L - tg**2 * bet1 * K
        nu31 = tg * (bet1 * k4 + l4)
        nu32 = k4 - tg**2 * bet1 * l4
        nu41 = tg * K
        nu42 = L
        nu61 = L
        nu62 = -tg * K
        nu71 = k4
        nu72 = -tg * l4
        nu81 = L - tg**2 * bet1 * K
        nu82 = -tg * (bet1 * L + K)
        nu101 = k4 * (1 - tg**2 * bet1**2) - 2 * tg**2 * bet1 * l4
        nu102 = -tg * (l4 * (1 - tg**2 * bet1**2) + 2 * bet1 * k4)
        nu111 = k2 * kl - 2 * tg**2 * k1 * l1 * l2
        nu112 = -tg * (l2 * kl + 2 * k1 * k2 * l1)
        nu141 = L
        nu142 = -tg * K
        nu151 = k2**2 - tg**2 * l2**2
        nu152 = -2 * tg * k2 * l2
        nu161 = k4 - tg**2 * bet1 * l4
        nu162 = -tg * (l4 + bet1 * k4)
        nu171 = kl * (k2 - tg**2 * bet1 * l2) - 2 * tg**2 * k1 * l1 * (l2 + bet1 * k2)
        nu172 = -tg * (2 * k1 * l1 * (k2 - tg**2 * bet1 * l2) + (l2 + bet1 * k2) * kl)
        nu181 = k1**4 - 6 * tg**2 * k1**2 * l1**2 + tg**4 * l1**4
        nu182 = -4 * tg * k1 * l1 * kl
        nu191 = L * (1 - tg**2 * bet1**2) - 2 * tg**2 * bet1 * K
        nu192 = -tg * (K * (1 - tg**2 * bet1**2) + 2 * bet1 * L)

        # nu of the K's
        nuK11 = 3 * nu11 - 2 * nu21
        nuK12 = 3 * nu12 - 2 * nu22
        nuK21 = 2 * nu31 - 2 * nu41
        nuK22 = 2 * nu32 - 2 * nu42
        nuK31 = nu101 - 3 * nu111 - nu81
        nuK32 = nu102 - 3 * nu112 - nu82
        nuK41 = 4 * nu141 - 3 * nu151 - nu161
        nuK42 = 4 * nu142 - 3 * nu152 - nu162
        nuK51 = 3 * nu171 - nu181 - nu191
        nuK52 = 3 * nu172 - nu182 - nu192
        nuK61 = nu61
        nuK62 = nu62
        nuK71 = nu71
        nuK72 = nu72

        # Final nu's
        nukur21 = (
            -nuK22
            + 2 * nuK61
            - 2 * (nuK71 + tg * bet1 * nuK72)
            - nuK41 * (alpha - 1) / (2 * alpha - 3)
        )
        nukur22 = (
            nuK21
            + 2 * nuK62
            - 2 * (nuK72 - tg * bet1 * nuK71)
            - nuK42 * (alpha - 1) / (2 * alpha - 3)
        )
        nukur31 = (
            6 * nuK11
            + 3 * (nuK21 + tg * bet1 * nuK22)
            - 2 * nuK32
            + 5 * (alpha - 1) * (tg * bet1 * nuK41 - nuK42) / (2 * alpha - 3)
        )
        nukur32 = (
            6 * nuK12
            + 3 * (nuK22 - tg * bet1 * nuK21)
            + 2 * nuK31
            + 5 * (alpha - 1) * (nuK41 + tg * bet1 * nuK42) / (2 * alpha - 3)
        )
        nukur41 = (
            nuK31
            + tg * bet1 * nuK32
            + (nuK41 * (1 - tg**2 * bet1**2) + 2 * nuK42 * tg * bet1)
            * (alpha - 1)
            / (2 * alpha - 3)
            + 3 * nuK51
        )
        nukur42 = (
            nuK32
            - tg * bet1 * nuK31
            + (nuK42 * (1 - tg**2 * bet1**2) - 2 * nuK41 * tg * bet1)
            * (alpha - 1)
            / (2 * alpha - 3)
            + 3 * nuK52
        )

        remember_four[idxlow] = k4 * varxlow**4 + (
            (tg * varxlow**3 * (l4 - bet1 * k4)) / (1 + tg**2 * bet1**2)
        ) * (tg * bet1 * varxlow + (1 - varxlow * H1) / (np.pi * fX))
        remember_four[idxlow] = remember_four[idxlow] - (
            alpha**2 * sig1**2 / (np.pi * fX)
        ) * (
            (varxlow**2 / 2) * (nukur21 * Hx2[:, 0] + nukur22 * Hx2[:, 1])
            + (alpha * sig1 * varxlow / 6) * (nukur31 * Hx3[:, 0] + nukur32 * Hx3[:, 1])
            + (alpha**2 * sig1**2 / 3) * (nukur41 * Hx4[:, 0] + nukur42 * Hx4[:, 1])
        )

        if len(idxupp) > 0:
            if np.abs(bet1) < 1:
                remember_four[idxupp] = (
                    varxupp**4
                    * (k4 + np.sign(varxupp) * l4)
                    / (1 + np.sign(varxupp) * bet1)
                )
            elif np.abs(bet1) == 1:
                remember_four[idxupp] = varxupp**4 * k4
            else:
                raise ValueError("Beta_1 not within [-1,1]")

    non_shifted_cond_moments = np.zeros((len(grid), max_order))

    non_shifted_cond_moments[:, 0] = remember_esp
    if max_order >= 2:
        non_shifted_cond_moments[:, 1] = remember_sec
    if max_order >= 3:
        non_shifted_cond_moments[:, 2] = remember_thir
    if max_order >= 4:
        non_shifted_cond_moments[:, 3] = remember_four

    return non_shifted_cond_moments
