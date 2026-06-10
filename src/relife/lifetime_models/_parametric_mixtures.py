"""K-component parametric mixture survival models."""

from __future__ import annotations

from typing import Any, Self, TypeAlias

import numpy as np
from numpy.typing import NDArray
from optype.numpy import Array1D, Array2D, ArrayND
from scipy.special import logsumexp
from typing_extensions import override

from relife.base import FittingResults
from relife.utils import to_column_2d_if_1d

from ._base import FittableParametricLifetimeModel, ParametricLifetimeModel
from ._distributions import LifetimeDistribution
from ._parametric_regressions import ParametricLifetimeRegression

__all__ = ["Mixture"]

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


class Mixture(ParametricLifetimeModel[*tuple[Any, ...]]):
    r"""K-component parametric mixture survival model.

    The survival function of the mixture is:

    .. math::

        S(t) = \sum_{k=1}^{K} \pi_k S_k(t)

    where :math:`\pi_k` are the mixing weights (:math:`\sum_k \pi_k = 1`,
    :math:`\pi_k > 0`) and :math:`S_k` are the component survival functions.

    Fitting is done via the EM algorithm, fully supporting right-censored (RC)
    and left-truncated right-censored (LTRC) data.  Components can be any
    homogeneous set of :class:`FittableParametricLifetimeModel` subclasses —
    both ``LifetimeDistribution`` and ``ParametricLifetimeRegression`` objects
    are accepted, provided all components share the same concrete family.  For
    regression components, covariates are forwarded as the first positional
    argument after ``time``.

    Parameters
    ----------
    *components : FittableParametricLifetimeModel
        At least two component models, all of the same concrete family
        (all ``LifetimeDistribution`` or all ``ParametricLifetimeRegression``).
    mix_weights : ndarray of shape (K,), optional
        Initial mixing weights.  Must be positive and sum to 1.  If not
        provided, uniform weights ``1/K`` are used.

    Examples
    --------
    Two-component Weibull mixture on right-censored data::

        model = Mixture(Weibull(), Weibull())
        model.fit(time, event=event)

    Two-component proportional-hazard mixture with covariates::

        model = Mixture(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
        )
        model.fit(time, covar, event=event)
    """

    fitting_results: FittingResults | None
    _mix_weights: NDArray[np.float64]

    def __init__(
        self,
        *components: FittableParametricLifetimeModel[*tuple[Any, ...]],
        mix_weights: NDArray[np.float64] | None = None,
    ) -> None:
        if len(components) < 2:
            raise ValueError(
                f"Mixture requires at least 2 components, got {len(components)}"
            )
        is_dist = all(isinstance(c, LifetimeDistribution) for c in components)
        is_reg = all(isinstance(c, ParametricLifetimeRegression) for c in components)
        if not (is_dist or is_reg):
            types = [type(c).__name__ for c in components]
            raise TypeError(
                "All components must belong to the same model family "
                "(all LifetimeDistribution or all ParametricLifetimeRegression). "
                f"Got {types}"
            )

        super().__init__()
        self.fitting_results = None

        K = len(components)
        for k, comp in enumerate(components):
            setattr(self, f"component_{k}", comp)

        self.mix_weights = mix_weights if mix_weights is not None else np.full(K, 1.0 / K)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nb_components(self) -> int:
        """Number of mixture components."""
        k = 0
        while hasattr(self, f"component_{k}"):
            k += 1
        return k

    @property
    def components(self) -> list[FittableParametricLifetimeModel[*tuple[Any, ...]]]:
        """List of component models in order."""
        return [getattr(self, f"component_{k}") for k in range(self.nb_components)]

    @property
    def mix_weights(self) -> NDArray[np.float64]:
        """Mixing weights, shape (K,)."""
        return self._mix_weights.copy()

    @mix_weights.setter
    def mix_weights(self, value: NDArray[np.float64]) -> None:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (self.nb_components,):
            raise ValueError(
                f"mix_weights must have shape ({self.nb_components},), got {value.shape}"
            )
        if not np.isclose(value.sum(), 1.0):
            raise ValueError(f"mix_weights must sum to 1, got sum = {value.sum():.6g}")
        if np.any(value <= 0):
            raise ValueError("All mix_weights must be strictly positive")
        self._mix_weights = value.copy()

    # ------------------------------------------------------------------
    # Core survival functions
    # ------------------------------------------------------------------

    @override
    def sf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: Any
    ) -> np.float64 | ArrayND[np.float64]:
        """Mixture survival function :math:`S(t) = \\sum_k \\pi_k S_k(t)`."""
        t = to_column_2d_if_1d(time) if args else time
        return sum(
            w * comp.sf(t, *args)
            for w, comp in zip(self._mix_weights, self.components)
        )

    @override
    def pdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: Any
    ) -> np.float64 | ArrayND[np.float64]:
        """Mixture density :math:`f(t) = \\sum_k \\pi_k f_k(t)`."""
        t = to_column_2d_if_1d(time) if args else time
        return sum(
            w * comp.pdf(t, *args)
            for w, comp in zip(self._mix_weights, self.components)
        )

    @override
    def hf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: Any
    ) -> np.float64 | ArrayND[np.float64]:
        """Mixture hazard function :math:`h(t) = f(t) / S(t)`."""
        return self.pdf(time, *args) / self.sf(time, *args)

    @override
    def chf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: Any
    ) -> np.float64 | ArrayND[np.float64]:
        """Mixture cumulative hazard :math:`H(t) = -\\log S(t)`."""
        return -np.log(self.sf(time, *args))

    # ------------------------------------------------------------------
    # Random variate sampling
    # ------------------------------------------------------------------

    @override
    def rvs(
        self,
        size: int | tuple[int, int],
        *args: Any,
        seed: (
            int
            | np.random.Generator
            | np.random.BitGenerator
            | np.random.RandomState
            | None
        ) = None,
    ) -> np.float64 | ArrayND[np.float64]:
        """Sample lifetimes using latent class assignment.

        Draws component membership from ``Categorical(π)``, then samples from
        each component.  For regression components the covariate rows are split
        by assignment; they must have a leading dimension equal to the total
        number of samples.

        Parameters
        ----------
        size : int or tuple (m, n)
            Number of samples, or ``(m, n)`` for ``m`` assets each with ``n``
            samples (output shape ``(m, n)``).
        *args
            Forwarded to each component model (e.g. covariate matrix for
            regression components).  For ``size=n`` the covariate must have
            shape ``(n, nb_coef)`` (one row per sample).  For
            ``size=(m, n)`` it must have shape ``(m, nb_coef)`` (one row per
            asset); it is internally expanded to ``(m*n, nb_coef)`` before
            sampling.
        seed : optional
            Seed for the random number generator.
        """
        rng = np.random.default_rng(seed)
        n_total = size if isinstance(size, int) else size[0] * size[1]

        # When size=(m, n), per-asset covar has shape (m, nb_coef). Expand to
        # (m*n, nb_coef) so every observation has its own row before masking.
        if isinstance(size, tuple):
            n_samples = size[1]
            args = tuple(
                np.repeat(np.atleast_2d(a), n_samples, axis=0)
                if np.asarray(a).ndim > 0 and np.asarray(a).shape[0] == size[0]
                else a
                for a in args
            )

        is_per_obs = bool(args) and any(
            np.asarray(a).ndim > 0 and np.asarray(a).shape[0] == n_total for a in args
        )

        component_indices = rng.choice(self.nb_components, size=n_total, p=self._mix_weights)
        times = np.empty(n_total)

        for k, comp in enumerate(self.components):
            mask = component_indices == k
            n_k = int(mask.sum())
            if n_k == 0:
                continue
            if is_per_obs:
                k_args = tuple(
                    np.atleast_2d(a)[mask]
                    if np.asarray(a).ndim > 0 and np.asarray(a).shape[0] == n_total
                    else a
                    for a in args
                )
                times[mask] = np.ravel(comp.rvs((n_k, 1), *k_args, seed=rng))
            else:
                times[mask] = np.ravel(comp.rvs(n_k, seed=rng))

        if isinstance(size, tuple):
            times = times.reshape(size)

        return times

    # ------------------------------------------------------------------
    # EM internals
    # ------------------------------------------------------------------

    def _log_likelihood(
        self,
        time: NDArray[np.float64],
        *args: Any,
        event: NDArray[np.bool_],
        entry: NDArray[np.float64],
    ) -> float:
        """Observed-data LTRC mixture log-likelihood."""
        mix_pdf = np.ravel(self.pdf(time, *args))
        mix_sf_time = np.ravel(self.sf(time, *args))
        log_num = np.where(
            event,
            np.log(np.maximum(mix_pdf, 1e-300)),
            np.log(np.maximum(mix_sf_time, 1e-300)),
        )
        log_denom = np.where(
            entry > 0,
            np.log(np.maximum(np.ravel(self.sf(entry, *args)), 1e-300)),
            0.0,
        )
        return float(np.sum(log_num - log_denom))

    def _estep(
        self,
        time: NDArray[np.float64],
        *args: Any,
        event: NDArray[np.bool_],
        entry: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """E-step: compute posterior component weights q[i, k].

        Uses log-sum-exp for numerical stability.  Implements the LTRC-aware
        formula: for uncensored observations the unnormalised log-weight is
        ``log π_k + log f_k(t) − log S_k(τ)``, for censored observations it
        is ``log π_k + log S_k(t) − log S_k(τ)``.
        """
        n = len(time)
        K = self.nb_components
        log_unnorm = np.empty((n, K))
        t = to_column_2d_if_1d(time) if args else time
        e = to_column_2d_if_1d(entry) if args else entry

        for k, (w, comp) in enumerate(zip(self._mix_weights, self.components)):
            log_sf_time = np.log(np.maximum(np.ravel(comp.sf(t, *args)), 1e-300))
            log_pdf_time = np.log(np.maximum(np.ravel(comp.pdf(t, *args)), 1e-300))
            log_sf_entry = np.where(
                entry > 0,
                np.log(np.maximum(np.ravel(comp.sf(e, *args)), 1e-300)),
                0.0,
            )
            log_unnorm[:, k] = (
                np.log(w)
                + np.where(event, log_pdf_time, log_sf_time)
                - log_sf_entry
            )

        log_norm = logsumexp(log_unnorm, axis=1, keepdims=True)
        return np.exp(log_unnorm - log_norm)

    def _mstep_weights(self, q: NDArray[np.float64]) -> None:
        """M-step: update mixing weights as the column-wise mean of q."""
        self._mix_weights = q.mean(axis=0)

    def _mstep_component(
        self,
        k: int,
        time: NDArray[np.float64],
        args: Array1D[Any] | Array2D[Any] | tuple[Array1D[Any] | Array2D[Any], ...] | None,
        *,
        event: NDArray[np.bool_],
        entry: NDArray[np.float64],
        weights_k: NDArray[np.float64],
        optimizer_options: dict[str, Any] | None,
    ) -> None:
        """M-step: update component k via its weighted LTRC likelihood."""
        if weights_k.sum() < 1e-10 or not np.all(np.isfinite(weights_k)):
            return
        comp = self.components[k]
        comp_opts = dict(optimizer_options) if optimizer_options else {}
        comp.fit(
            time,
            args,
            event=event,
            entry=entry if np.any(entry > 0) else None,
            weights=weights_k,
            **comp_opts,
        )

    def _init_components(
        self,
        time: NDArray[np.float64],
        *args: Any,
        event: NDArray[np.bool_],
    ) -> None:
        """Initialise component parameters from quantile bands of the data.

        Fits each component on a distinct quantile band to break symmetry.
        Uses simple RC likelihood (no left-truncation) for speed.  Falls back
        to a scale estimate derived from the band's median lifetime when the
        band fit fails or produces extreme parameters.
        """
        n = len(time)
        K = self.nb_components
        sorted_idx = np.argsort(time)
        band_edges = np.linspace(0, n, K + 1, dtype=int)
        fallback_scales = np.percentile(time, np.linspace(5, 95, K))

        for k, comp in enumerate(self.components):
            band_idx = sorted_idx[band_edges[k] : band_edges[k + 1]]
            if len(band_idx) < 5:
                band_idx = sorted_idx
            k_time = time[band_idx]
            k_event = event[band_idx]
            k_args = tuple(
                np.asarray(a).reshape(n, -1)[band_idx]
                if np.ndim(a) > 0 and np.shape(a)[0] == n
                else a
                for a in args
            )
            try:
                comp.fit(k_time, *k_args, event=k_event)
                if not np.all(np.isfinite(comp.get_params())) or np.any(
                    np.abs(comp.get_params()) > 1e4
                ):
                    raise ValueError("extreme params after band fit")
            except Exception:
                nb_params = comp.get_params().size
                comp.set_params(np.full(nb_params, 1.0 / fallback_scales[k]))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        time: Array1D[np.float64],
        *args: Any,
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        max_iter: int = 300,
        tol: float = 1e-6,
        optimizer_options: dict[str, Any] | None = None,
    ) -> Self:
        """Fit the mixture model via the EM algorithm.

        Supports right-censored (RC) and left-truncated right-censored (LTRC)
        data.  For regression components, pass the covariate matrix as the
        first positional argument after ``time``.

        Parameters
        ----------
        time : 1-D array
            Observed lifetimes (or censoring times).
        *args :
            Extra positional arguments forwarded to each component model.
            For regression components this should be the covariate matrix.
        event : 1-D bool array, optional
            ``True`` for observed events, ``False`` for right-censored
            observations.  Defaults to all-``True``.
        entry : 1-D float array, optional
            Left-truncation times.  Defaults to all-zero (no truncation).
        max_iter : int, default 300
            Maximum number of EM iterations.
        tol : float, default 1e-6
            Convergence tolerance on the relative change of the observed-data
            log-likelihood: ``|ΔL| / (1 + |L|) < tol``.
        optimizer_options : dict, optional
            Extra keyword arguments forwarded to ``scipy.optimize.minimize``
            for each component's M-step.

        Returns
        -------
        self
        """
        time = np.ravel(np.asarray(time, dtype=np.float64))
        n = len(time)
        event = (
            np.ones(n, dtype=np.bool_)
            if event is None
            else np.ravel(np.asarray(event, dtype=np.bool_))
        )
        entry = (
            np.zeros(n, dtype=np.float64)
            if entry is None
            else np.ravel(np.asarray(entry, dtype=np.float64))
        )

        K = self.nb_components

        # --- Initialisation ---
        if any(np.any(np.isnan(comp.get_params())) for comp in self.components):
            self._init_components(time, *args, event=event)

        # --- EM loop ---
        prev_ll = self._log_likelihood(time, *args, event=event, entry=entry)
        converged = False

        for _ in range(max_iter):
            q = self._estep(time, *args, event=event, entry=entry)
            self._mstep_weights(q)
            for k in range(K):
                self._mstep_component(
                    k,
                    time,
                    args[0] if args else None,
                    event=event,
                    entry=entry,
                    weights_k=q[:, k],
                    optimizer_options=optimizer_options,
                )
            ll = self._log_likelihood(time, *args, event=event, entry=entry)
            if abs(ll - prev_ll) / (1.0 + abs(ll)) < tol:
                converged = True
                break
            prev_ll = ll

        # --- Store results ---
        ll_final = self._log_likelihood(time, *args, event=event, entry=entry)
        self.fitting_results = FittingResults(
            nb_observations=n,
            optimal_params=np.concatenate([self._mix_weights[:-1], self.get_params()]),
            success=converged,
            neg_log_likelihood=-ll_final,
            covariance_matrix=None,
        )
        return self
