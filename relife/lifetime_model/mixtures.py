from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from relife.likelihood import DefaultLifetimeLikelihood, FittingResults

from ._base import ParametricLifetimeModel
from .distribution import LifetimeDistribution
from .regression import LifetimeRegression

__all__ = ["Mixture"]


class Mixture(ParametricLifetimeModel):
    r"""K-component parametric mixture survival model.

    The survival function of the mixture is:

    .. math::

        S(t) = \sum_{k=1}^{K} \pi_k S_k(t)

    where :math:`\pi_k` are the mixing weights (:math:`\sum_k \pi_k = 1`,
    :math:`\pi_k > 0`) and :math:`S_k` are the component survival functions.

    Fitting is done via the EM algorithm, fully supporting right-censored (RC)
    and left-truncated right-censored (LTRC) data.  Components can be any
    homogeneous set of :class:`ParametricLifetimeModel` subclasses — both
    ``LifetimeDistribution`` and ``LifetimeRegression`` objects are accepted,
    provided all components share the same concrete type.  For regression
    components, covariates are forwarded as positional ``*args`` to every
    method and to ``fit``.

    Parameters
    ----------
    *components : ParametricLifetimeModel
        At least two component models, all of the same concrete type.
    weights : ndarray of shape (K,), optional
        Initial mixing weights.  Must be positive and sum to 1.  If not
        provided, uniform weights ``1/K`` are used.

    Examples
    --------
    Two-component Weibull mixture on right-censored data::

        model = Mixture(Weibull(), Weibull())
        model.fit(time, event=event)

    Two-component proportional-hazard mixture with covariates::

        model = Mixture(ProportionalHazard(Weibull()), ProportionalHazard(Weibull()))
        model.fit(time, covar, event=event)
    """

    def __init__(
        self,
        *components: ParametricLifetimeModel,
        weights: NDArray[np.float64] | None = None,
    ):
        if len(components) < 2:
            raise ValueError(
                f"Mixture requires at least 2 components, got {len(components)}"
            )
        # Components must all belong to the same family: either all
        # LifetimeDistribution or all LifetimeRegression.  Mixing families
        # would produce an incoherent model (different calling conventions).
        is_dist = all(isinstance(c, LifetimeDistribution) for c in components)
        is_reg = all(isinstance(c, LifetimeRegression) for c in components)
        if not (is_dist or is_reg):
            types = [type(c).__name__ for c in components]
            raise TypeError(
                "All components must belong to the same model family "
                "(all LifetimeDistribution or all LifetimeRegression). "
                f"Got {types}"
            )

        super().__init__()

        K = len(components)
        for k, comp in enumerate(components):
            setattr(self, f"component_{k}", comp)

        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != (K,):
                raise ValueError(
                    f"weights must have shape ({K},), got {weights.shape}"
                )
            if not np.isclose(weights.sum(), 1.0):
                raise ValueError(
                    f"weights must sum to 1, got sum = {weights.sum():.6g}"
                )
            if np.any(weights <= 0):
                raise ValueError("All weights must be strictly positive")
            self._weights = weights.copy()
        else:
            self._weights = np.full(K, 1.0 / K)

        self.fitting_results: FittingResults | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nb_components(self) -> int:
        """Number of mixture components."""
        return sum(1 for k in self._baseline_models if k.startswith("component_"))

    @property
    def components(self) -> list[ParametricLifetimeModel]:
        """List of component models in order."""
        return [self._baseline_models[f"component_{k}"] for k in range(self.nb_components)]

    @property
    def weights(self) -> NDArray[np.float64]:
        """Mixing weights, shape (K,)."""
        return self._weights.copy()

    @weights.setter
    def weights(self, value: NDArray[np.float64]) -> None:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (self.nb_components,):
            raise ValueError(
                f"weights must have shape ({self.nb_components},), got {value.shape}"
            )
        if not np.isclose(value.sum(), 1.0):
            raise ValueError(f"weights must sum to 1, got sum = {value.sum():.6g}")
        if np.any(value <= 0):
            raise ValueError("All weights must be strictly positive")
        self._weights = value.copy()

    # ------------------------------------------------------------------
    # Core survival functions  (abstract methods of ParametricLifetimeModel)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_col(time):
        """Reshape time to (n, 1) for regression component calls."""
        return np.asarray(time).reshape(-1, 1)

    def sf(self, time, *args):
        """Mixture survival function :math:`S(t) = \\sum_k \\pi_k S_k(t)`."""
        t = self._to_col(time) if args else time
        return sum(
            w * comp.sf(t, *args)
            for w, comp in zip(self._weights, self.components)
        )

    def pdf(self, time, *args):
        """Mixture probability density :math:`f(t) = \\sum_k \\pi_k f_k(t)`."""
        t = self._to_col(time) if args else time
        return sum(
            w * comp.pdf(t, *args)
            for w, comp in zip(self._weights, self.components)
        )

    def hf(self, time, *args):
        """Mixture hazard function :math:`h(t) = f(t) / S(t)`."""
        return self.pdf(time, *args) / self.sf(time, *args)

    def chf(self, time, *args):
        """Mixture cumulative hazard :math:`H(t) = -\\log S(t)`."""
        return -np.log(self.sf(time, *args))

    # ------------------------------------------------------------------
    # Random variate sampling
    # ------------------------------------------------------------------

    def rvs(
        self,
        size,
        *args,
        nb_assets=None,
        return_event=False,
        return_entry=False,
        seed=None,
    ):
        """Sample lifetimes using latent class assignment.

        Draws component membership from ``Categorical(π)``, then samples from
        each component.  For regression components the ``args`` array(s) are
        split by component assignment; they must have a leading dimension equal
        to the total number of samples requested.
        """
        rng = np.random.default_rng(seed)
        K = self.nb_components
        n_total = size if nb_assets is None else nb_assets * size

        # Detect whether args are per-observation (regression case): at least one
        # arg has a leading dimension equal to n_total.
        is_per_obs = args and any(
            np.asarray(a).ndim > 0 and np.asarray(a).shape[0] == n_total
            for a in args
        )

        component_indices = rng.choice(K, size=n_total, p=self._weights)
        times = np.empty(n_total)

        for k, comp in enumerate(self.components):
            mask = component_indices == k
            n_k = int(mask.sum())
            if n_k == 0:
                continue
            if is_per_obs:
                # Regression: subset covariate rows for this component, then call
                # rvs(1, covar_k) which produces one lifetime per row (n_k, 1).
                k_args = tuple(
                    np.atleast_2d(a)[mask]
                    if np.asarray(a).ndim > 0 and np.asarray(a).shape[0] == n_total
                    else a
                    for a in args
                )
                times[mask] = np.ravel(comp.rvs(1, *k_args, seed=rng))
            else:
                # Distribution: sample n_k i.i.d. lifetimes.
                times[mask] = np.ravel(comp.rvs(n_k, seed=rng))

        if nb_assets is not None:
            times = times.reshape(nb_assets, size)

        event = np.ones_like(times, dtype=np.bool_)
        entry = np.zeros_like(times, dtype=np.float64)

        if not return_event and not return_entry:
            return times
        elif return_event and not return_entry:
            return times, event
        elif not return_event and return_entry:
            return times, entry
        else:
            return times, event, entry

    # ------------------------------------------------------------------
    # EM fitting
    # ------------------------------------------------------------------

    def _log_likelihood(self, time, *args, event, entry) -> float:
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

    def _estep(self, time, *args, event, entry) -> NDArray[np.float64]:
        """E-step: compute posterior component weights q[i, k].

        Uses log-sum-exp for numerical stability.  Implements the LTRC-aware
        formula: for uncensored observations the unnormalised log-weight is
        ``log π_k + log f_k(y) − log S_k(τ)``, for censored observations it
        is ``log π_k + log S_k(y) − log S_k(τ)``.
        """
        n = len(time)
        K = self.nb_components
        log_unnorm = np.empty((n, K))
        t = self._to_col(time) if args else time
        e = self._to_col(entry) if args else entry

        for k, (w, comp) in enumerate(zip(self._weights, self.components)):
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

        log_norm = logsumexp(log_unnorm, axis=1, keepdims=True)  # (n, 1)
        return np.exp(log_unnorm - log_norm)  # (n, K)

    def _mstep_weights(self, q: NDArray[np.float64]) -> None:
        """M-step: update mixing weights as the column-wise mean of q."""
        self._weights = q.mean(axis=0)

    def _mstep_component(
        self,
        k: int,
        time,
        *args,
        event,
        entry,
        weights_k: NDArray[np.float64],
        optimizer_options: dict | None,
    ) -> None:
        """M-step: update component k by minimising its weighted LTRC likelihood."""
        if weights_k.sum() < 1e-10:
            return  # degenerate component: skip
        comp = self.components[k]
        comp_opts = dict(optimizer_options) if optimizer_options else {}
        if "bounds" not in comp_opts:
            comp_opts["bounds"] = comp._get_params_bounds()
        likelihood = DefaultLifetimeLikelihood(
            comp, time, *args, event=event, entry=entry, weights=weights_k
        )
        result = likelihood.maximum_likelihood_estimation(**comp_opts)
        comp.params = result.optimal_params

    def fit(
        self,
        time,
        *args,
        event=None,
        entry=None,
        max_iter: int = 300,
        tol: float = 1e-6,
        optimizer_options: dict | None = None,
    ):
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
        # If any component has uninitialised params, fit each component on a
        # different quantile band of the observed times using simple RC
        # likelihood (no left-truncation correction).  Using the full LTRC
        # likelihood on a small subset is numerically fragile; the EM loop
        # will use the proper LTRC likelihood from the first iteration onwards.
        # Fitting with RC on different time bands breaks symmetry and gives
        # each component a distinct, meaningful starting point.
        needs_init = any(np.any(np.isnan(comp.params)) for comp in self.components)
        if needs_init:
            sorted_idx = np.argsort(time)
            band_edges = np.linspace(0, n, K + 1, dtype=int)
            # Quantile grid used as fallback if band fitting fails
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
                    # RC-only fit for initialization (entry omitted intentionally)
                    comp.fit(k_time, *k_args, event=k_event)
                    if not np.all(np.isfinite(comp.params)) or np.any(
                        np.abs(comp.params) > 1e4
                    ):
                        raise ValueError("extreme params after band fit")
                except Exception:
                    # Fallback: param-scale initialization via _get_initial_params
                    comp._get_initial_params(
                        np.full(len(time), fallback_scales[k]), event=event
                    )

        # --- EM loop ---
        prev_ll = self._log_likelihood(time, *args, event=event, entry=entry)

        for _ in range(max_iter):
            q = self._estep(time, *args, event=event, entry=entry)       # (n, K)
            self._mstep_weights(q)
            for k in range(K):
                self._mstep_component(
                    k, time, *args,
                    event=event, entry=entry,
                    weights_k=q[:, k],
                    optimizer_options=optimizer_options,
                )
            ll = self._log_likelihood(time, *args, event=event, entry=entry)
            if abs(ll - prev_ll) / (1.0 + abs(ll)) < tol:
                break
            prev_ll = ll

        # --- Store results ---
        ll_final = self._log_likelihood(time, *args, event=event, entry=entry)
        self.fitting_results = FittingResults(
            nb_obversations=n,
            optimal_params=np.concatenate([self._weights[:-1], self.params]),
            neg_log_likelihood=-ll_final,
            covariance_matrix=None,
        )
        return self
