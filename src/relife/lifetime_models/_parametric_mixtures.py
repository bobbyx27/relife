"""K-component parametric mixture survival models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self, TypeAlias

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from optype.numpy import Array1D, Array2D, ArrayND
from scipy.special import logsumexp
from typing_extensions import override

from relife.base import FittingResults, ParametricModel
from relife.utils import to_column_2d_if_1d

from ._base import FittableParametricLifetimeModel, LifetimeData, ParametricLifetimeModel
from ._distributions import LifetimeDistribution, init_distrib_params_from_lifetimes
from ._parametric_regressions import (
    ParametricLifetimeRegression,
    init_regression_params_from_lifetimes,
)

__all__ = [
    "ParametricLifetimeMixture",
    "ParametricLifetimeMixtureWithWeightsRegression",
]

_MIN_WEIGHT_SUM: float = 1e-10
_MIN_BAND_SIZE: int = 5

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


class MixtureWeightsRegression(ParametricModel):
    """Softmax regression model for mixture component weights.

    Maps covariates to per-sample component probabilities via a linear layer
    followed by a softmax activation.  Parameters are fitted by minimising
    cross-entropy loss against the E-step posterior responsibilities.

    Parameters
    ----------
    nb_coef : int
        Number of input covariates.
    nb_components : int
        Number of mixture components (output classes ``K``).

    Notes
    -----
    The underlying :class:`torch.nn.Linear` module is always kept on CPU.
    After fitting, parameters are synchronised back into the ``ParametricModel``
    ``_params`` tree so that :meth:`get_params` reflects the fitted values.
    """

    torch_module: nn.Linear

    def __init__(self, nb_coef: int, nb_components: int) -> None:
        # nn.Linear is not a ParametricModel so __setattr__ just stores it normally;
        # safe to set before super().__init__() since _baseline_models is never touched.
        self.torch_module = nn.Linear(nb_coef, nb_components)
        weight_vals = self.torch_module.weight.data.numpy().ravel().tolist()
        bias_vals = self.torch_module.bias.data.numpy().ravel().tolist()
        super().__init__(
            **{f"weight_{i}": v for i, v in enumerate(weight_vals)},
            **{f"bias_{i}": v for i, v in enumerate(bias_vals)},
        )

    def _sync_params(self) -> None:
        """Copy fitted torch parameter values into ``_params``."""
        params = np.concatenate([
            self.torch_module.weight.data.numpy().ravel(),
            self.torch_module.bias.data.numpy().ravel(),
        ]).astype(np.float64)
        self.set_params(params)

    def predict(self, covar: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return per-sample component probabilities, shape ``(n, K)``.

        Parameters
        ----------
        covar : ndarray of shape ``(n, nb_coef)``
        """
        X = torch.tensor(np.asarray(covar, dtype=np.float32))
        with torch.no_grad():
            return torch.softmax(self.torch_module(X), dim=-1).numpy().astype(np.float64)

    def fit(
        self,
        covar: NDArray[np.float64],
        q: NDArray[np.float64],
        *,
        max_iter: int = 1000,
        lr: float = 1e-3,
    ) -> Self:
        """Fit via full-batch gradient descent on cross-entropy loss.

        Parameters
        ----------
        covar : ndarray of shape ``(n, nb_coef)``
        q : ndarray of shape ``(n, K)``
            Soft target weights (E-step posterior responsibilities).
        max_iter : int
            Number of gradient steps.
        lr : float
            Learning rate for Adam.
        """
        X = torch.tensor(np.asarray(covar, dtype=np.float32))
        targets = torch.tensor(np.asarray(q, dtype=np.float32))
        optimizer = torch.optim.Adam(self.torch_module.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(max_iter):
            optimizer.zero_grad()
            loss_fn(self.torch_module(X), targets).backward()
            optimizer.step()
        self._sync_params()
        return self

    def __call__(self, covar: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.predict(covar)


class _FittableParametricLifetimeModelMixture(
    ParametricLifetimeModel[*tuple[Any, ...]], ABC
):
    r"""Abstract base for K-component parametric mixture survival models.

    The survival function of the mixture is:

    .. math::

        S(t \mid x) = \sum_{k=1}^{K} \pi_k(x)\, S_k(t \mid x)

    where :math:`\pi_k(x)` are the mixing weights and :math:`S_k` the
    component survival functions.

    Subclasses must implement :meth:`_get_weights` — which controls how
    mixing weights are computed (fixed scalars or covariate-dependent
    regression) — and :meth:`_mstep_weights`, the corresponding M-step
    update.  All EM mechanics, component management, and survival functions
    are provided here.

    Parameters
    ----------
    *components : FittableParametricLifetimeModel
        At least two component models, all of the same concrete family
        (all :class:`LifetimeDistribution` or all
        :class:`ParametricLifetimeRegression`).

    Notes
    -----
    Dynamically-named component attributes ``component_0``, ``component_1``,
    … cannot be declared statically; they are the acknowledged exception to
    the class-level attribute declaration convention.
    """

    fitting_results: FittingResults | None

    def __init__(
        self,
        *components: FittableParametricLifetimeModel[*tuple[Any, ...]],
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
        for k, comp in enumerate(components):
            setattr(self, f"component_{k}", comp)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_weights(
        self, covar: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        """Return mixing weights.

        Returns
        -------
        ndarray
            Shape ``(K,)`` for fixed weights or ``(n, K)`` for per-sample
            regression weights where ``n = covar.shape[0]``.
        """

    @abstractmethod
    def _mstep_weights(
        self,
        q: NDArray[np.float64],
        covar: NDArray[np.float64] | None = None,
    ) -> None:
        """M-step: update mixing weights from posterior responsibilities ``q``."""

    # ------------------------------------------------------------------
    # FittingResults hook (non-abstract, override in fixed-weight subclass)
    # ------------------------------------------------------------------

    def _fitted_mix_params(self) -> NDArray[np.float64]:
        """Extra mixing parameters prepended to ``FittingResults.optimal_params``.

        Returns an empty array by default: used when the weights model
        parameters are already registered in ``_params`` (i.e.
        :class:`ParametricLifetimeMixtureWithWeightsRegression`).
        :class:`ParametricLifetimeMixture` overrides this to return
        ``_mix_weights[:-1]``.
        """
        return np.array([], dtype=np.float64)

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

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _w_k(
        w: NDArray[np.float64], k: int
    ) -> np.float64 | NDArray[np.float64]:
        """Per-component weight slice safe against the ``(n,) × (n,1)`` broadcast bug.

        Returns ``w[k]`` (scalar) for 1-D weight arrays and ``w[:, k:k+1]``
        (column vector) for 2-D per-sample weight arrays, so that
        multiplication with a component output of shape ``(n, 1)`` always
        yields ``(n, 1)`` rather than ``(n, n)``.
        """
        return w[k] if w.ndim == 1 else w[:, k : k + 1]

    @staticmethod
    def _sample_component_indices(
        rng: np.random.Generator,
        n_total: int,
        nb_components: int,
        p: NDArray[np.float64],
    ) -> NDArray[np.intp]:
        """Draw component indices supporting ``(K,)`` or per-sample ``(n, K)`` weights."""
        if p.ndim == 1:
            return rng.choice(nb_components, size=n_total, p=p)
        return np.array([rng.choice(nb_components, p=p[i] / p[i].sum()) for i in range(n_total)])

    # ------------------------------------------------------------------
    # Core survival functions
    # ------------------------------------------------------------------

    @override
    def sf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: Any
    ) -> np.float64 | ArrayND[np.float64]:
        """Mixture survival function :math:`S(t) = \\sum_k \\pi_k S_k(t)`."""
        t = to_column_2d_if_1d(time) if args else time
        w = self._get_weights(args[0] if args else None)
        return sum(
            self._w_k(w, k) * comp.sf(t, *args)
            for k, comp in enumerate(self.components)
        )

    @override
    def pdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: Any
    ) -> np.float64 | ArrayND[np.float64]:
        """Mixture density :math:`f(t) = \\sum_k \\pi_k f_k(t)`."""
        t = to_column_2d_if_1d(time) if args else time
        w = self._get_weights(args[0] if args else None)
        return sum(
            self._w_k(w, k) * comp.pdf(t, *args)
            for k, comp in enumerate(self.components)
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

        Draws component membership from ``Categorical(π)`` — per-sample when
        weights have shape ``(n, K)``, global when ``(K,)`` — then samples from
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

        component_indices = self._sample_component_indices(
            rng,
            n_total,
            self.nb_components,
            self._get_weights(args[0] if args else None),
        )
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
            np.log(np.maximum(mix_pdf, np.finfo(float).tiny)),
            np.log(np.maximum(mix_sf_time, np.finfo(float).tiny)),
        )
        log_denom = np.where(
            entry > 0,
            np.log(np.maximum(np.ravel(self.sf(entry, *args)), np.finfo(float).tiny)),
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

        w = self._get_weights(args[0] if args else None)  # (K,) or (n, K)
        for k, comp in enumerate(self.components):
            log_sf_time = np.log(np.maximum(np.ravel(comp.sf(t, *args)), np.finfo(float).tiny))
            log_pdf_time = np.log(np.maximum(np.ravel(comp.pdf(t, *args)), np.finfo(float).tiny))
            log_sf_entry = np.where(
                entry > 0,
                np.log(np.maximum(np.ravel(comp.sf(e, *args)), np.finfo(float).tiny)),
                0.0,
            )
            log_unnorm[:, k] = (
                np.log(np.maximum(w[..., k], np.finfo(float).tiny))
                + np.where(event, log_pdf_time, log_sf_time)
                - log_sf_entry
            )

        log_norm = logsumexp(log_unnorm, axis=1, keepdims=True)
        return np.exp(log_unnorm - log_norm)

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
        if weights_k.sum() < _MIN_WEIGHT_SUM or not np.all(np.isfinite(weights_k)):
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
        args: Array1D[Any] | Array2D[Any] | tuple[Array1D[Any] | Array2D[Any], ...] | None,
        *,
        event: NDArray[np.bool_],
        entry: NDArray[np.float64],
    ) -> None:
        """Initialise component parameters from quantile bands of the data.

        Fits each component on a distinct quantile band to break symmetry.
        Falls back to principled parameter estimates from the band data when
        the band fit fails or produces non-finite parameters.
        """
        n = len(time)
        K = self.nb_components
        sorted_idx = np.argsort(time)
        band_edges = np.linspace(0, n, K + 1, dtype=int)

        for k, comp in enumerate(self.components):
            band_idx = sorted_idx[band_edges[k] : band_edges[k + 1]]
            if len(band_idx) < _MIN_BAND_SIZE:
                band_idx = sorted_idx
            k_time = time[band_idx]
            k_event = event[band_idx]
            k_entry = entry[band_idx]
            k_args = (
                np.asarray(args).reshape(n, -1)[band_idx]
                if args is not None and np.ndim(args) > 0 and np.shape(args)[0] == n
                else args
            )
            k_entry_arg = k_entry if np.any(k_entry > 0) else None
            try:
                comp.fit(k_time, k_args, event=k_event, entry=k_entry_arg)
                if not np.all(np.isfinite(comp.get_params())):
                    raise ValueError("non-finite params after band fit")
            except Exception:
                band_data = LifetimeData(k_time, k_args, event=k_event, entry=k_entry_arg)
                if isinstance(comp, LifetimeDistribution):
                    param0 = init_distrib_params_from_lifetimes(comp, band_data)
                else:
                    param0 = init_regression_params_from_lifetimes(comp, band_data)
                comp.set_params(param0)

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

        if any(np.any(np.isnan(comp.get_params())) for comp in self.components):
            self._init_components(time, args[0] if args else None, event=event, entry=entry)

        ll = self._log_likelihood(time, *args, event=event, entry=entry)
        converged = False

        for _ in range(max_iter):
            q = self._estep(time, *args, event=event, entry=entry)
            self._mstep_weights(q, covar=args[0] if args else None)
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
            prev_ll, ll = ll, self._log_likelihood(time, *args, event=event, entry=entry)
            if abs(ll - prev_ll) / (1.0 + abs(ll)) < tol:
                converged = True
                break

        self.fitting_results = FittingResults(
            nb_observations=n,
            optimal_params=np.concatenate([self._fitted_mix_params(), self.get_params()]),
            success=converged,
            neg_log_likelihood=-ll,
            covariance_matrix=None,
        )
        return self


class ParametricLifetimeMixture(_FittableParametricLifetimeModelMixture):
    r"""K-component parametric mixture with fixed mixing weights.

    The survival function of the mixture is:

    .. math::

        S(t) = \sum_{k=1}^{K} \pi_k S_k(t)

    where :math:`\pi_k` are constant mixing weights (:math:`\sum_k \pi_k = 1`,
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

        model = ParametricLifetimeMixture(Weibull(), Weibull())
        model.fit(time, event=event)

    Two-component proportional-hazard mixture with covariates::

        model = ParametricLifetimeMixture(
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
        super().__init__(*components)
        K = self.nb_components
        self.mix_weights = mix_weights if mix_weights is not None else np.full(K, 1.0 / K)

    # ------------------------------------------------------------------
    # mix_weights property
    # ------------------------------------------------------------------

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
    # Abstract method implementations
    # ------------------------------------------------------------------

    @override
    def _get_weights(self, covar: NDArray[np.float64] | None) -> NDArray[np.float64]:
        return self._mix_weights  # (K,)

    @override
    def _mstep_weights(
        self,
        q: NDArray[np.float64],
        covar: NDArray[np.float64] | None = None,
    ) -> None:
        self._mix_weights = q.mean(axis=0)

    @override
    def _fitted_mix_params(self) -> NDArray[np.float64]:
        return self._mix_weights[:-1]


class ParametricLifetimeMixtureWithWeightsRegression(
    _FittableParametricLifetimeModelMixture
):
    r"""K-component parametric mixture with covariate-dependent mixing weights.

    Mixing weights are modelled as a softmax regression:

    .. math::

        \pi_k(x) = \frac{\exp(w_k^\top x + b_k)}{\sum_{j} \exp(w_j^\top x + b_j)}

    so that each observation ``i`` contributes to the mixture with its own
    per-sample weight vector :math:`\pi(x_i) \in \Delta^{K-1}`.

    Parameters
    ----------
    *components : FittableParametricLifetimeModel
        At least two component models, all of the same concrete family
        (all ``LifetimeDistribution`` or all ``ParametricLifetimeRegression``).
    nb_coef : int
        Number of covariates fed to the weights regression model.

    Examples
    --------
    Two-component proportional-hazard mixture where weights depend on covar::

        model = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=3,
        )
        model.fit(time, covar, event=event)
    """

    fitting_results: FittingResults | None
    weights_model: MixtureWeightsRegression

    def __init__(
        self,
        *components: FittableParametricLifetimeModel[*tuple[Any, ...]],
        nb_coef: int,
    ) -> None:
        super().__init__(*components)
        self.weights_model = MixtureWeightsRegression(nb_coef, self.nb_components)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    @override
    def _get_weights(self, covar: NDArray[np.float64] | None) -> NDArray[np.float64]:
        return self.weights_model.predict(covar)  # (n, K)

    @override
    def _mstep_weights(
        self,
        q: NDArray[np.float64],
        covar: NDArray[np.float64] | None = None,
    ) -> None:
        self.weights_model.fit(covar, q)
