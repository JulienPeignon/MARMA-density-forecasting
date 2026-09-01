"""
=================
ispline class obtained from: https://jbloomlab.github.io/dms_variants/_modules/dms_variants/ispline.html#Isplines
pdf and cdf fitting funtions added by Biprateep Dey
=================

Implements :class:`Isplines`, which are monotonic spline functions that are
defined in terms of :class:`Msplines`.

See `Ramsay (1988)`_ for details about these splines, and also note the
corrections in the `Praat manual`_ to the errors in the I-spline formula
by `Ramsay (1988)`_.

.. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395
.. _`Praat manual`: http://www.fon.hum.uva.nl/praat/manual/spline.html

"""

import numpy as np
from sklearn.linear_model import LinearRegression


def fit_cdf(x, y, x_predict=None, num_basis=10, fit_intercept=False):
    """Least-squares fit of a monotone I-spline CDF to ``(x, y)`` on [0, 1].

    Parameters
    ----------
    x, y : np.ndarray
        Abscissae in [0, 1] and the CDF values to interpolate.
    x_predict : np.ndarray or None
        Where to evaluate the fit; ``x`` when None.
    num_basis : int
        Number of I-spline basis functions. ``num_basis - 1`` mesh points, so
        values <= 3 give no interior knot at all (a single quadratic).
    fit_intercept : bool
        Kept for backwards compatibility; leave False.

    Returns
    -------
    (y_fit, coef, intercept)

    Raises
    ------
    ValueError
        If ``num_basis`` is not an integer above ``order - 2``.
    """
    order = 3  # fixing the I-spline order
    num_mesh_points = num_basis + 2 - order  # num_splines = num_mesh_points + 2 - order
    if not isinstance(num_basis, int) or (num_mesh_points <= 0):
        raise ValueError(f"num_basis should be an integer greater than {order - 2}")

    mesh = np.linspace(0, 1, num_mesh_points)
    isplines = Isplines(order, mesh, x)

    #     if fit_intercept:
    #         num_basis = num_basis + 1
    X = np.ones((len(x), num_basis))
    for i in range(isplines.n):
        X[:, i] = isplines.I(i + 1)
    model = LinearRegression(positive=True, fit_intercept=fit_intercept)
    model.fit(X, y)
    if not fit_intercept:
        total = model.coef_.sum()
        if np.isfinite(total) and total > 0:
            model.coef_ = model.coef_ / total
    if x_predict is not None:
        isplines = Isplines(order, mesh, x_predict)
        X = np.ones((len(x_predict), num_basis))
        for i in range(isplines.n):
            X[:, i] = isplines.I(i + 1)
        y_fit = model.predict(X)
    else:
        y_fit = model.predict(X)

    return y_fit, model.coef_, model.intercept_


def get_pdf(cdf_grid, cdf, pdf_grid, num_basis=10, fit_intercept=False):
    """Density of the monotone I-spline CDF fitted to ``(cdf_grid, cdf)``.

    Returns
    -------
    (pdf, coef, max_abs_residual)
    """
    y_fit, coef, _ = fit_cdf(
        x=cdf_grid, y=cdf, num_basis=num_basis, fit_intercept=fit_intercept
    )
    residual = float(np.max(np.abs(y_fit - cdf)))

    order = 3  # fixing the I-spline order
    num_mesh_points = num_basis + 2 - order  # num_splines = num_mesh_points + 2 - order
    mesh = np.linspace(0, 1, num_mesh_points)
    pdf_grid = np.minimum(np.asarray(pdf_grid, dtype=float), 1.0 - 1e-12)
    isplines = Isplines(order, mesh, pdf_grid)

    pdf = np.ones((len(pdf_grid), (isplines.n)))
    for i in range(isplines.n):
        pdf[:, i] = isplines.dI_dx(i + 1)
    norm = coef.sum()  # The basis CDF range from 0 to 1, i.e. basis PDFs integrate to 1
    if not np.isfinite(norm) or norm <= 0:
        return np.ones(len(pdf_grid)), coef, residual
    pdf = np.sum(coef * pdf, axis=-1) / norm

    return pdf, coef, residual


class Isplines:
    r"""Implements I-splines (see `Ramsay (1988)`_).

    Parameters
    ----------
    order : int
        Sets :attr:`Isplines.order`.
    mesh : array-like
        Sets :attr:`Isplines.mesh`.
    x : np.ndarray
        Sets :attr:`Isplines.x`.

    Attributes
    ----------
    order : int
        Order of spline, :math:`k` in notation of `Ramsay (1988)`_. Note that
        the degree of the I-spline is equal to :math:`k`, while the
        associated M-spline has order :math:`k` but degree :math:`k - 1`.
    mesh : np.ndarray
        Mesh sequence, :math:`\xi_1 < \ldots < \xi_q` in the notation
        of `Ramsay (1988)`_. This class implements **fixed** mesh sequences.
    n : int
        Number of members in spline, denoted as :math:`n` in `Ramsay (1988)`_.
        Related to number of points :math:`q` in the mesh and the order
        :math:`k` by :math:`n = q - 2 + k`.
    lower : float
        Lower end of interval spanned by the splines (first point in mesh).
    upper : float
        Upper end of interval spanned by the splines (last point in mesh).

    Note
    ----
    The methods of this class cache their results and return immutable
    numpy arrays. Do **not** make these arrays mutable and change their
    values, as this will lead to invalid caching.

    Example
    -------
    Short examples to demonstrate and test :class:`Isplines`:

    .. plot::
       :context: reset

       >>> import itertools
       >>> import numpy as np
       >>> import pandas as pd
       >>> import scipy.optimize
       >>> from dms_variants.ispline import Isplines

       >>> order = 3
       >>> mesh = [0.0, 0.3, 0.5, 0.6, 1.0]
       >>> x = np.array([0, 0.2, 0.3, 0.4, 0.8, 0.99999])
       >>> isplines = Isplines(order, mesh, x)
       >>> isplines.order
       3
       >>> isplines.mesh
       array([0. , 0.3, 0.5, 0.6, 1. ])
       >>> isplines.n
       6
       >>> isplines.lower
       0.0
       >>> isplines.upper
       1.0

       Evaluate the I-splines at some selected points:

       >>> for i in range(1, isplines.n + 1):
       ...     print(f"I{i}: {np.round(isplines.I(i), 2)}")
       ... # doctest: +NORMALIZE_WHITESPACE
       I1: [0.   0.96 1.   1.   1.   1.  ]
       I2: [0.   0.52 0.84 0.98 1.   1.  ]
       I3: [0.   0.09 0.3  0.66 1.   1.  ]
       I4: [0.   0.   0.   0.02 0.94 1.  ]
       I5: [0.   0.   0.   0.   0.58 1.  ]
       I6: [0.   0.   0.   0.   0.13 1.  ]

       Check that gradients are correct for :meth:`Isplines.dI_dx`:

       >>> for i, xval in itertools.product(range(1, isplines.n + 1), x):
       ...     xval = np.array([xval])
       ...     def func(xval):
       ...         return Isplines(order, mesh, xval).I(i)
       ...     def dfunc(xval):
       ...         return Isplines(order, mesh, xval).dI_dx(i)
       ...     err = scipy.optimize.check_grad(func, dfunc, xval)
       ...     if err > 1e-5:
       ...         raise ValueError(f"excess err {err} for {i}, {xval}")

       Plot the I-splines in Fig. 1 of `Ramsay (1988)`_:

       >>> xplot = np.linspace(0, 1, 1000)
       >>> isplines_xplot = Isplines(order, mesh, xplot)
       >>> data = {'x': xplot}
       >>> for i in range(1, isplines.n + 1):
       ...     data[f"I{i}"] = isplines_xplot.I(i)
       >>> df = pd.DataFrame(data)
       >>> _ = df.plot(x='x')

    .. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395

    """

    def __init__(self, order, mesh, x):
        """See main class docstring."""
        if not (isinstance(order, int) and order >= 1):
            raise ValueError(f"`order` not int >= 1: {order}")
        self.order = order

        self.mesh = np.array(mesh, dtype="float")
        if self.mesh.ndim != 1:
            raise ValueError(f"`mesh` not array-like of dimension 1: {mesh}")
        if len(self.mesh) < 2:
            raise ValueError(f"`mesh` not length >= 2: {mesh}")
        if not np.array_equal(self.mesh, np.unique(self.mesh)):
            raise ValueError(f"`mesh` elements not unique and sorted: {mesh}")
        self.lower = self.mesh[0]
        self.upper = self.mesh[-1]
        assert self.lower < self.upper

        self.n = len(self.mesh) - 2 + self.order

        if not (isinstance(x, np.ndarray) and x.ndim == 1):
            raise ValueError("`x` is not np.ndarray of dimension 1")
        if (x < self.lower).any() or (x > self.upper).any():
            raise ValueError(f"`x` outside {self.lower} and {self.upper}: {x}")
        self._x = x.copy()
        self._x.flags.writeable = False

        self._msplines = Msplines(order + 1, mesh, self.x)

        # for caching values
        self._cache = {}
        self._max_cache_size = 100

    @property
    def x(self):
        """np.ndarray: Points at which spline is evaluated."""
        return self._x

    def I(self, i):  # noqa: E743,E741
        r"""Evaluate spline :math:`I_i` at point(s) :attr:`Isplines.x`.

        Parameters
        ----------
        i : int
            Spline member :math:`I_i`, where :math:`1 \le i \le`
            :attr:`Isplines.n`.

        Returns
        -------
        np.ndarray
            The values of the I-spline at each point in :attr:`Isplines.x`.

        Note
        ----
        The spline is evaluated using the formula given in the
        `Praat manual`_, which corrects some errors in the formula
        provided by `Ramsay (1988)`_:

        .. math::

           I_i\left(x\right)
           =
           \begin{cases}
           0 & \rm{if\;} i > j, \\
           1 & \rm{if\;} i < j - k, \\
           \sum_{m=i+1}^j \left(t_{m+k+1} - t_m\right)
                          M_m\left(x \mid k + 1\right) / \left(k + 1 \right)
             & \rm{otherwise},
           \end{cases}

        where :math:`j` is the index such that :math:`t_j \le x < t_{j+1}`
        (the :math:`\left\{t_j\right\}` are the :attr:`Msplines.knots` for a
        M-spline of order :math:`k + 1`) and :math:`k` is
        :attr:`Isplines.order`.

        .. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395
        .. _`Praat manual`: http://www.fon.hum.uva.nl/praat/manual/spline.html

        """
        args = (i, "I")
        if args not in self._cache:
            if len(self._cache) > self._max_cache_size:
                self._cache = {}
            self._cache[args] = self._calculate_I_or_dI(*args)
        return self._cache[args]

    @property
    def j(self):
        """np.ndarray: :math:`j` as defined in :meth:`Isplines.I`."""
        if not hasattr(self, "_j"):
            self._j = np.searchsorted(self._msplines.knots, self.x, "right")
            assert (1 <= self._j).all() and (self._j <= len(self._msplines.knots)).all()
            assert self.x.shape == self._j.shape
        return self._j

    @property
    def _sum_terms_I(self):
        """np.ndarray: sum terms for :meth:`Isplines.I`.

        Row `m - 1` has summation term for `m`.

        """
        if not hasattr(self, "_sum_terms_I_val"):
            k = self.order
            self._sum_terms_I_val = np.vstack(
                [
                    (self._msplines.knots[m + k] - self._msplines.knots[m - 1])
                    * self._msplines.M(m, k + 1)
                    / (k + 1)
                    for m in range(1, self._msplines.n + 1)
                ]
            )
            assert self._sum_terms_I_val.shape == (self._msplines.n, len(self.x))
        return self._sum_terms_I_val

    @property
    def _sum_terms_dI_dx(self):
        """np.ndarray: sum terms for :meth:`Isplines.dI_dx`.

        Row `m - 1` has summation term for `m`.

        """
        if not hasattr(self, "_sum_terms_dI_dx_val"):
            k = self.order
            self._sum_terms_dI_dx_val = np.vstack(
                [
                    (self._msplines.knots[m + k] - self._msplines.knots[m - 1])
                    * self._msplines.dM_dx(m, k + 1)
                    / (k + 1)
                    for m in range(1, self._msplines.n + 1)
                ]
            )
            assert self._sum_terms_dI_dx_val.shape == (self._msplines.n, len(self.x))
        return self._sum_terms_dI_dx_val

    def _calculate_I_or_dI(self, i, quantity):
        """Calculate :meth:`Isplines.I` or :meth:`Isplines.dI_dx`.

        Parameters
        ----------
        i : int
            Same meaning as for :meth:`Isplines.I`.
        quantity : {'I', 'dI'}
            Calculate :meth:`Isplines.I` or :meth:`Isplines.dI_dx`?

        Returns
        -------
        np.ndarray
            The return value of :meth:`Isplines.I` or :meth:`Isplines.dI_dx`.

        Note
        ----
        Most calculations for :meth:`Isplines.I` and :meth:`Isplines.dI_dx`
        are the same, so this method implements both.

        """
        if quantity == "I":
            sum_terms = self._sum_terms_I
            i_lt_jminusk = 1.0
        elif quantity == "dI":
            sum_terms = self._sum_terms_dI_dx
            i_lt_jminusk = 0.0
        else:
            raise ValueError(f"invalid `quantity` {quantity}")

        if not (1 <= i <= self.n):
            raise ValueError(f"invalid spline member `i` of {i}")

        k = self.order

        # create `binary_terms` where entry (m - 1, x) is 1 if and only if
        # the corresponding `sum_terms` entry is part of the sum.
        binary_terms = np.vstack(
            [
                np.zeros(len(self.x)) if m < i + 1 else (m <= self.j).astype(int)
                for m in range(1, self._msplines.n + 1)
            ]
        )
        assert binary_terms.shape == sum_terms.shape

        # compute sums from `sum_terms` and `binary_terms`
        sums = np.sum(sum_terms * binary_terms, axis=0)
        assert sums.shape == self.x.shape

        # return value with sums, 0, or 1
        res = np.where(i > self.j, 0.0, np.where(i < self.j - k, i_lt_jminusk, sums))
        res.flags.writeable = False
        return res

    def dI_dx(self, i):
        r"""Derivative of :meth:`Isplines.I` by :attr:`Isplines.x`.

        Parameters
        ----------
        i : int
            Same meaning as for :meth:`Isplines.I`.

        Returns
        -------
        np.ndarray
            Derivative of I-spline with respect to :attr:`Isplines.x`.

        Note
        ----
        The derivative is calculated from the equation in :meth:`Isplines.I`:

        .. math::

           \frac{\partial I_i\left(x\right)}{\partial x}
           =
           \begin{cases}
           0 & \rm{if\;} i > j \rm{\; or \;} i < j - k, \\
           \sum_{m=i+1}^j\left(t_{m+k+1} - t_m\right)
                         \frac{\partial M_m\left(x \mid k+1\right)}{\partial x}
                         \frac{1}{k + 1}
             & \rm{otherwise}.
           \end{cases}

        """
        args = (i, "dI")
        if args not in self._cache:
            if len(self._cache) > self._max_cache_size:
                self._cache = {}
            self._cache[args] = self._calculate_I_or_dI(*args)
        return self._cache[args]


class Msplines:
    r"""Implements M-splines (see `Ramsay (1988)`_).

    Parameters
    ----------
    order : int
        Sets :attr:`Msplines.order`.
    mesh : array-like
        Sets :attr:`Msplines.mesh`.
    x : np.ndarray
        Sets :attr:`Msplines.x`.

    Attributes
    ----------
    order : int
        Order of spline, :math:`k` in notation of `Ramsay (1988)`_.
        Polynomials are of degree :math:`k - 1`.
    mesh : np.ndarray
        Mesh sequence, :math:`\xi_1 < \ldots < \xi_q` in the notation
        of `Ramsay (1988)`_. This class implements **fixed** mesh sequences.
    n : int
        Number of members in spline, denoted as :math:`n` in `Ramsay (1988)`_.
        Related to number of points :math:`q` in the mesh and the order
        :math:`k` by :math:`n = q - 2 + k`.
    knots : np.ndarray
        The knot sequence, :math:`t_1, \ldots, t_{n + k}` in the notation of
        `Ramsay (1988)`_.
    lower : float
        Lower end of interval spanned by the splines (first point in mesh).
    upper : float
        Upper end of interval spanned by the splines (last point in mesh).

    Note
    ----
    The methods of this class cache their results and return immutable
    numpy arrays. Do **not** make those arrays mutable and change their
    values as this will lead to invalid caching.

    Example
    -------
    Demonstrate and test :class:`Msplines`:

    .. plot::
       :context: reset

       >>> import functools
       >>> import itertools
       >>> import numpy as np
       >>> import pandas as pd
       >>> import scipy.optimize
       >>> from dms_variants.ispline import Msplines

       >>> order = 3
       >>> mesh = [0.0, 0.3, 0.5, 0.6, 1.0]
       >>> x = np.array([0, 0.2, 0.3, 0.4, 0.8, 0.99999])
       >>> msplines = Msplines(order, mesh, x)
       >>> msplines.order
       3
       >>> msplines.mesh
       array([0. , 0.3, 0.5, 0.6, 1. ])
       >>> msplines.n
       6
       >>> msplines.knots
       array([0. , 0. , 0. , 0.3, 0.5, 0.6, 1. , 1. , 1. ])
       >>> msplines.lower
       0.0
       >>> msplines.upper
       1.0

       Evaluate the M-splines at some selected points:

       >>> for i in range(1, msplines.n + 1):
       ...     print(f"M{i}: {np.round(msplines.M(i), 2)}")
       ... # doctest: +NORMALIZE_WHITESPACE
       M1: [10. 1.11 0.  0.   0.   0.  ]
       M2: [0.  3.73 2.4 0.6  0.   0.  ]
       M3: [0.  1.33 3.  3.67 0.   0.  ]
       M4: [0.  0.   0.  0.71 0.86 0.  ]
       M5: [0.  0.   0.  0.   3.3  0.  ]
       M6: [0.  0.   0.  0.   1.88 7.5 ]

       Check that the gradients are correct:

       >>> for i, xval in itertools.product(range(1, msplines.n + 1), x):
       ...     xval = np.array([xval])
       ...     def func(xval):
       ...         return Msplines(order, mesh, xval).M(i)
       ...     def dfunc(xval):
       ...         return Msplines(order, mesh, xval).dM_dx(i)
       ...     err = scipy.optimize.check_grad(func, dfunc, xval)
       ...     if err > 1e-5:
       ...         raise ValueError(f"excess err {err} for {i}, {xval}")

       Plot the M-splines in in Fig. 1 of `Ramsay (1988)`_:

       >>> xplot = np.linspace(0, 1, 1000, endpoint=False)
       >>> msplines_plot = Msplines(order, mesh, xplot)
       >>> data = {'x': xplot}
       >>> for i in range(1, msplines_plot.n + 1):
       ...     data[f"M{i}"] = msplines_plot.M(i)
       >>> df = pd.DataFrame(data)
       >>> _ = df.plot(x='x')

    .. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395

    """

    def __init__(self, order, mesh, x):
        """See main class docstring."""
        if not (isinstance(order, int) and order >= 1):
            raise ValueError(f"`order` not int >= 1: {order}")
        self.order = order

        self.mesh = np.array(mesh, dtype="float")
        if self.mesh.ndim != 1:
            raise ValueError(f"`mesh` not array-like of dimension 1: {mesh}")
        if len(self.mesh) < 2:
            raise ValueError(f"`mesh` not length >= 2: {mesh}")
        if not np.array_equal(self.mesh, np.unique(self.mesh)):
            raise ValueError(f"`mesh` elements not unique and sorted: {mesh}")
        self.lower = self.mesh[0]
        self.upper = self.mesh[-1]
        assert self.lower < self.upper

        self.knots = np.array(
            [self.lower] * self.order
            + list(self.mesh[1:-1])
            + [self.upper] * self.order,
            dtype="float",
        )

        self.n = len(self.knots) - self.order
        assert self.n == len(self.mesh) - 2 + self.order

        if not (isinstance(x, np.ndarray) and x.ndim == 1):
            raise ValueError("`x` is not np.ndarray of dimension 1")
        if (x < self.lower).any() or (x > self.upper).any():
            raise ValueError(f"`x` outside {self.lower} and {self.upper}: {x}")
        self._x = x.copy()
        self._x.flags.writeable = False

        self._ti_le_x_lt_tiplusk_cache = {}

        # for caching values
        self._M_cache = {}
        self._dM_cache = {}
        self._max_cache_size = 100

    def _ti_le_x_lt_tiplusk(self, ti, tiplusk):
        r"""Indices where :math:`t_i \le x \le t_{i+k}`.

        Parameters
        ----------
        ti : float
            :math:`t_i`
        tiplusk : float
            :math:`t_{i+k}`

        Returns
        -------
        np.ndarray
            Array of booleans of same length as :attr:`Msplines.x` indicating
            if :math:`t_i \le x \le t_{i+k}`.

        """
        key = (ti, tiplusk)
        if key not in self._ti_le_x_lt_tiplusk_cache:
            val = (ti <= self.x) & (self.x < tiplusk)
            val.flags.writeable = False
            assert val.dtype == bool
            if len(self._ti_le_x_lt_tiplusk_cache) > self._max_cache_size:
                self._ti_le_x_lt_tiplusk_cache = {}
            self._ti_le_x_lt_tiplusk_cache[key] = val
        return self._ti_le_x_lt_tiplusk_cache[key]

    @property
    def x(self):
        """np.ndarray: Points at which spline is evaluated."""
        return self._x

    def M(self, i, k=None, invalid_i="raise"):
        r"""Evaluate spline :math:`M_i` at point(s) :attr:`Msplines.x`.

        Parameters
        ----------
        i : int
            Spline member :math:`M_i`, where :math:`1 \le i \le`
            :attr:`Msplines.n`.
        k : int or None
            Order of spline. If `None`, assumed to be :attr:`Msplines.order`.
        invalid_i : {'raise', 'zero'}
            If `i` is invalid, do we raise an error or return 0?

        Returns
        -------
        np.ndarray
            The values of the M-spline at each point in :attr:`Msplines.x`.

        Note
        ----
        The spline is evaluated using the recursive relationship given by
        `Ramsay (1988) <https://www.jstor.org/stable/2245395>`_:

        .. math::

           M_i\left(x \mid k=1\right)
           &=&
           \begin{cases}
           1 / \left(t_{i+1} - t_i\right), & \rm{if\;} t_i \le x < t_{i+1} \\
           0, & \rm{otherwise}
           \end{cases} \\
           M_i\left(x \mid k > 1\right) &=&
           \begin{cases}
           \frac{k\left[\left(x - t_i\right) M_i\left(x \mid k-1\right) +
                        \left(t_{i+k} -x\right) M_{i+1}\left(x \mid k-1\right)
                        \right]}
           {\left(k - 1\right)\left(t_{i + k} - t_i\right)},
           & \rm{if\;} t_i \le x < t_{i+k} \\
           0, & \rm{otherwise}
           \end{cases}

        """
        args = (i, k, invalid_i)
        if args not in self._M_cache:
            if len(self._M_cache) > self._max_cache_size:
                self._M_cache = {}
            self._M_cache[args] = self._calculate_M(*args)
        return self._M_cache[args]

    def _calculate_M(self, i, k, invalid_i):
        """Calculate :meth:`Msplines.M` with result caching."""
        if not (1 <= i <= self.n):
            if invalid_i == "raise":
                raise ValueError(f"invalid spline member `i` of {i}")
            elif invalid_i == "zero":
                return 0
            else:
                raise ValueError(f"invalid `invalid_i` of {invalid_i}")
        if k is None:
            k = self.order
        if not 1 <= k <= self.order:
            raise ValueError(f"invalid spline order `k` of {k}")

        tiplusk = self.knots[i + k - 1]
        ti = self.knots[i - 1]
        if tiplusk == ti:
            return 0

        boolindex = self._ti_le_x_lt_tiplusk(ti, tiplusk)
        if k == 1:
            res = np.where(boolindex, 1.0 / (tiplusk - ti), 0.0)
            res.flags.writeable = False
            return res
        else:
            assert k > 1
            res = np.where(
                boolindex,
                (
                    k
                    * (
                        (self.x - ti) * self.M(i, k - 1)
                        + (tiplusk - self.x) * self.M(i + 1, k - 1, invalid_i="zero")
                    )
                    / ((k - 1) * (tiplusk - ti))
                ),
                0.0,
            )
            res.flags.writeable = False
            return res

    def dM_dx(self, i, k=None, invalid_i="raise"):
        r"""Derivative of :meth:`Msplines.M` by to :attr:`Msplines.x`.

        Parameters
        ----------
        i : int
            Same as for :meth:`Msplines.M`.
        k : int or None
            Same as for :meth:`Msplines.M`.
        invalid_i : {'raise', 'zero'}
            Same as for :meth:`Msplines.M`.

        Returns
        -------
        np.ndarray
            Derivative of M-spline with respect to :attr:`Msplines.x`.

        Note
        ----
        The derivative is calculated from the equation in :meth:`Msplines.M`:

        .. math::

           \frac{\partial M_i\left(x \mid k=1\right)}{\partial x} &=& 0
           \\
           \frac{\partial M_i\left(x \mid k > 1\right)}{\partial x}
           &=&
           \begin{cases}
           \frac{k\left[\left(x - t_i\right)
                        \frac{\partial M_i\left(x \mid k-1\right)}{\partial x}
                        +
                        M_i\left(x \mid k-1\right)
                        +
                        \left(t_{i+k} -x\right)
                        \frac{\partial M_{i+1}\left(x \mid k-1\right)}
                             {\partial x}
                        -
                        M_{i+1}\left(x \mid k-1\right)
                        \right]}
           {\left(k - 1\right)\left(t_{i + k} - t_i\right)},
           & \rm{if\;} t_i \le x < t_{i+1} \\
           0, & \rm{otherwise}
           \end{cases}

        """
        args = (i, k, invalid_i)
        if args not in self._dM_cache:
            if len(self._dM_cache) > self._max_cache_size:
                self._dM_cache = {}
            self._dM_cache[args] = self._calculate_dM_dx(*args)
        return self._dM_cache[args]

    def _calculate_dM_dx(self, i, k, invalid_i):
        """Calculate :meth:`Msplines.dM_dx` with results caching."""
        if not (1 <= i <= self.n):
            if invalid_i == "raise":
                raise ValueError(f"invalid spline member `i` of {i}")
            elif invalid_i == "zero":
                return 0
            else:
                raise ValueError(f"invalid `invalid_i` of {invalid_i}")
        if k is None:
            k = self.order
        if not 1 <= k <= self.order:
            raise ValueError(f"invalid spline order `k` of {k}")

        tiplusk = self.knots[i + k - 1]
        ti = self.knots[i - 1]
        if tiplusk == ti or k == 1:
            return 0
        else:
            assert k > 1
            boolindex = self._ti_le_x_lt_tiplusk(ti, tiplusk)
            res = np.where(
                boolindex,
                (
                    k
                    * (
                        (self.x - ti) * self.dM_dx(i, k - 1)
                        + self.M(i, k - 1)
                        + (tiplusk - self.x)
                        * self.dM_dx(i + 1, k - 1, invalid_i="zero")
                        - self.M(i + 1, k - 1, invalid_i="zero")
                    )
                    / ((k - 1) * (tiplusk - ti))
                ),
                0.0,
            )
            res.flags.writeable = False
            return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
