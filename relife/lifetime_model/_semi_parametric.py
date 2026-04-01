import copy
from dataclasses import dataclass, field

import numpy as np
from numpy._typing import NDArray
from optype.numpy import Array1D, ToFloat, ToFloat2D
from scipy.optimize import Bounds, minimize
from scipy.stats import norm
from scipy.special import erf
from numba import njit, prange, float64
from typing_extensions import Literal, Unpack, final, overload, override, Callable

from relife.base import MaximumLikelihoodFittingResults, MaximumLikehoodOptimizer
from relife.lifetime_model._regression import LinearCovarEffect
from relife.typing import MaximumLikelihoodOptimizerOptions, ScipyMinimizeOptions
from relife.utils import reshape_1d_arg


@dataclass
class CoxData:
    time: NDArray[np.float64]
    covar: NDArray[np.float64]
    event: NDArray[np.bool_] | None = None
    entry: NDArray[np.float64] | None = None
    likelihood_to_use: Literal["cox"] | Literal["breslow"] | Literal["efron"] = field(
        init=False, repr=True
    )
    ordered_event_time: NDArray[np.float64] = field(init=False, repr=False)
    event_count: NDArray[np.int64] = field(init=False, repr=False)
    risk_set: NDArray[np.bool_] = field(init=False, repr=False)
    death_set: NDArray[np.bool_] = field(init=False, repr=False)
    ordered_event_covar: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        self.time = reshape_1d_arg(self.time)
        self.event = (
            reshape_1d_arg(self.event)
            if self.event is not None
            else np.ones_like(self.time, dtype=np.bool_)
        )
        self.entry = (
            reshape_1d_arg(self.entry)
            if self.entry is not None
            else np.zeros_like(self.time, dtype=np.float64)
        )
        sizes = [len(x) for x in (self.time, self.event, self.entry, self.covar)]

        if len(set(sizes)) != 1:
            raise ValueError(
                f""""
                All lifetime data must have the same number of values. Fields
                length are different. Got {tuple(sizes)}.
                """
            )
        (
            self.ordered_event_time,  # uncensored sorted untied times
            ordered_event_index,
            self.event_count,
        ) = np.unique(
            self.time[self.event == 1],
            return_index=True,
            return_counts=True,
        )
        # here risk_set is mask array on time
        # left truncated & right censored
        self.risk_set = np.logical_and(
            (
                np.vstack([self.entry[:, 0]] * len(self.ordered_event_time))
                < np.hstack([self.ordered_event_time[:, None]] * len(self.time))
            ),
            (
                np.hstack([self.ordered_event_time[:, None]] * len(self.time))
                <= np.vstack([self.time[:, 0]] * len(self.ordered_event_time))
            ),
        )

        self.death_set = np.vstack(
            [self.time[:, 0] * self.event[:, 0]] * len(self.ordered_event_time)
        ) == np.hstack([self.ordered_event_time[:, None]] * len(self.time))

        self.ordered_event_covar = self.covar[self.event[:, 0] == 1][
            ordered_event_index
        ]


def psi(
    covar_effect: LinearCovarEffect,
    data: CoxData,
    on: Literal["risk"] | Literal["death"] = "risk",
    order: Literal[0] | Literal[1] | Literal[2] = 0,
) -> NDArray[np.float64]:
    """Psi formula used for likelihood computations

    Args:
        on (str, optional): "risk" or "death". Defaults to "risk". If "death",
        sum is applied on death set. order (int, optional): order derivatives
        with respect to params. Defaults to 0.

    Returns:
        np.ndarray: psi formulation
        If order 0, shape [m, 1]
        If order 1, shape [m, p]
        If order 2, shape [m, p, p]
    """
    if on == "risk":
        i_set = data.risk_set
    elif on == "death":
        i_set = data.death_set

    if order == 0:
        # shape [m]
        return np.dot(i_set, covar_effect.g(data.covar))
    elif order == 1:
        # shape [m, p]
        return np.dot(i_set, data.covar * covar_effect.g(data.covar))
    elif order == 2:
        # shape [m, p, p]
        return np.tensordot(
            i_set[:, :None],
            data.covar[:, None]
            * data.covar[:, :, None]
            * np.asarray(covar_effect.g(data.covar))[:, :, None],
            axes=1,
        ).astype(np.float64)


class _BreslowBaseline:
    """
    Class for Cox non-parametric Breslow baseline
    """

    data: CoxData
    covar_effect: LinearCovarEffect

    def __init__(self,
                 covar_effect: LinearCovarEffect,
                 data: CoxData
                 ):
        assert data.covar.shape[-1] == covar_effect.nb_params
        self.covar_effect = covar_effect
        self.data = data

    @overload
    def chf(
        self, se: Literal[False], kp: bool = False
    ) -> NDArray[np.float64]: ...
    @overload
    def chf(
        self, se: Literal[True], kp: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def chf(
        self, se: bool = False, kp: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | NDArray[np.float64]:
        """
        The cumulative hazard function estimation

        Parameters
        ----------
        se : bool, default is False
            If true, the estimated standard errors are returned too.

        Returns
        -------
        tuple of 2 or 3 ndarrays
            A tuple containing the timeline,
            the estimated values and optionally the estimated standard errors (if se is set to true)
        """
        # TODO : comprendre le kp
        if kp:
            values = np.cumsum(
                1
                - (
                    1
                    - (
                        self.covar_effect.g(self.data.ordered_event_covar)
                        / psi(self.covar_effect, self.data)
                    )
                )
                ** (self.covar_effect.g(self.data.ordered_event_covar))
            )
        else:
            values = np.cumsum(
                self.data.event_count[:, None] / psi(self.covar_effect, self.data)
            )
        if se:
            var = np.cumsum(
                self.data.event_count[:, None] / psi(self.covar_effect, self.data) ** 2
            )
            conf_int_values = np.hstack(
                [
                    values[:, None]
                    + np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                    values[:, None]
                    - np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                ]
            )
            return values, conf_int_values
        else:
            return values

    @overload
    def sf(self, se: Literal[False]) -> NDArray[np.float64]: ...
    @overload
    def sf(
        self, se: Literal[True]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def sf(
        self, se: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | NDArray[np.float64]:
        """
        The survival function estimation

        Parameters
        ----------
        se : bool, default is False
            If true, the estimated standard errors are returned too.

        Returns
        -------
        tuple of 2 or 3 ndarrays
            A tuple containing the timeline,
            the estimated values and optionally the estimated standard errors (if se is set to true)
        """
        if se:
            chf, chf_conf_int_values = self.chf(se=True)
            return np.exp(-chf), np.exp(-chf_conf_int_values)
        else:
            return np.exp(-self.chf(se=False))


class SemiParametricProportionalHazard:
    """
    Class for Cox, semi-parametric, Proportional Hazards, model
    """

    fitting_results: MaximumLikelihoodFittingResults | None
    covar_effect: LinearCovarEffect | None
    _training_data: CoxData | None
    _sf: NDArray[np.void] | None

    def __init__(self):
        self._sf = None
        self._training_data = None
        self.fitting_results = None
        self.covar_effect = None

    @property
    def params(self):
        if self.covar_effect is None:
            return None
        return self.covar_effect.params

    @property
    def nb_params(self):
        if self.covar_effect is None:
            return None
        return self.covar_effect.nb_params

    @overload
    def sf(
        self, se: Literal[False]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def sf(
        self, se: Literal[True]
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None
    ): ...
    @overload
    def sf(
        self, se: bool
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        | None
    ): ...

    def sf(
        self, covar: NDArray[np.float64], se: bool = True
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        | None
    ):
        """
        The survival function estimations.

        Parameters
        ----------
        covar: np.array
            array with covariates values
        se : bool, default True
            If True, the standard errors are returned in addition to timeline
            and sf values.

        Returns
        -------
        out : tuple of timeline, values, optionally se. Default is None
            A timeline, corresponding sf values and optionnaly the standard
            errors. If the estimations does not exist yet, returns None.
        """
        if self._sf is None:
            return None
        if se:
            return (
                self._sf["timeline"],
                self._sf["baseline_estimation"] ** self.covar_effect.g(covar),
                self._sf["baseline_estimation"] ** self.covar_effect.g(covar) * np.sqrt(self._q1_q2_sum(covar))
            )
        return (
            self._sf["timeline"],
            self._sf["baseline_estimation"] ** self.covar_effect.g(covar)
        )

    def _q1_q2_sum(
            self,
            covar: NDArray[np.float64]
    ) -> NDArray[np.float64] | None:
        """Voir Klein and Moeschberger: Survival Analysis Techniques for Censored and Truncated Data (p 284)"""
        # TODO: Is it somehow dependent to Breslow estimator ? I don't think so, but the reference isn't clear enough.
        # TODO: Faut-il en faire une fonction et non une méthode ?
        # TODO: estimation foireuse...
        if self.fitting_results is None:
            raise ValueError("This method can only be called in other methods after model fit")
        if self.fitting_results.covariance_matrix is not None:
            psi_values = psi(self.covar_effect, self._training_data)
            psi_order_1 = psi(self.covar_effect, self._training_data, order=1)
            d_j_on_psi = self._training_data.event_count[:, None] / psi_values

            q3 = np.cumsum(
                (psi_order_1 / psi_values)[None, :, :] - covar[:, None, :]
                * d_j_on_psi[None, :, :], axis=1
            )  # [m: new sample for inference, t: timeline, p]
            q2 = np.squeeze(
                np.matmul(
                    q3[:, :, None, :],
                    np.matmul(
                        self.fitting_results.covariance_matrix[None, None, :, :],
                        q3[:, :, :, None],
                    ),
                )
            )  # [m, t]
            q1 = np.cumsum(d_j_on_psi * (1 / psi_values))
            return q1 + q2
        return None

    def fit(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
        **optimizer_options: Unpack[MaximumLikelihoodOptimizerOptions],
    ):
        # init covar_effect
        self.covar_effect = LinearCovarEffect(
            (None,) * np.atleast_2d(np.asarray(covar, dtype=np.float64)).shape[-1]
        )

        _,  event_count = np.unique(time[event == 1], return_counts=True)
        if (event_count > 3).any():  # efron
            likelihood = EfronPartialLifetimeLikelihood(self.covar_effect, time, covar, event, entry)
        elif (event_count <= 3).all() and (2 in event_count):  # breslow
            likelihood = BreslowPartialLifetimeLikelihood(self.covar_effect, time, covar, event, entry)
        else:
            likelihood = CoxPartialLifetimeLikelihood(self.covar_effect, time, covar, event, entry)

        if "x0" not in optimizer_options:
            np.random.seed(1)
            optimizer_options["x0"] = np.random.random(covar.shape[1])
        if "method" not in optimizer_options:
            optimizer_options["method"] = "trust-exact"
        optimizer_options["jac"] = likelihood.jac_negative_log
        optimizer_options["hess"] = likelihood.hess_negative_log

        self.fitting_results = likelihood.maximum_likelihood_estimation(**optimizer_options)
        self.covar_effect.params = self.fitting_results.optimal_params
        self._training_data = likelihood.data

        # currently only BreslowBaseline is used to compute sf
        baseline = _BreslowBaseline(self.covar_effect, likelihood.data)

        timeline = likelihood.data.ordered_event_time
        dtype = np.dtype(
            [("timeline", np.float64), ("baseline_estimation", np.float64)]
        )
        self._sf = np.empty((timeline.size,), dtype=dtype)
        self._sf["timeline"] = timeline
        self._sf["baseline_estimation"] = baseline.sf(se=False)

        return self


@final
class CoxPartialLifetimeLikelihood(
    MaximumLikehoodOptimizer[LinearCovarEffect, CoxData]
):
    data: CoxData
    # https://github.com/microsoft/pyright/issues/6564
    model: LinearCovarEffect
    scipy_method = "trust-exact"

    def __init__(
        self,
        covar_effect: LinearCovarEffect,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
    ):
        self.model = copy.deepcopy(covar_effect)
        self.data = CoxData(time, covar, event=event, entry=entry)

    @property
    @override
    def nb_observations(self) -> int:
        return len(self.data.time)

    @override
    def _initialize_model(self) -> LinearCovarEffect:
        self.model.params = np.random.random(self.data.covar.shape[1])
        return self.model

    @override
    def _get_params_bounds(self) -> Bounds:
        return Bounds(
            np.full(self.model.nb_params, -np.inf),
            np.full(self.model.nb_params, np.inf),
        )

    @override
    def negative_log(self, params: Array1D[np.float64]) -> ToFloat:
        self.model.params = params
        return -(
            np.log(self.model.g(self.data.ordered_event_covar)).sum()
            - np.log(psi(self.model, self.data)).sum()
        )

    def jac_negative_log(self, params: Array1D[np.float64]) -> Array1D[np.float64]:
        self.model.params = params  # changes model params

        return -(
            self.data.ordered_event_covar.sum(axis=0)
            - (psi(self.model, self.data, order=1) / psi(self.model, self.data)).sum(
                axis=0
            )
        )

    def hess_negative_log(self, params: Array1D[np.float64]) -> ToFloat2D:
        self.model.params = params  # changes model params

        psi_order_0 = psi(self.model, self.data)
        psi_order_1 = psi(self.model, self.data, order=1)

        hessian_part_1 = psi(self.model, self.data, order=2) / psi_order_0[:, :, None]
        # print("hessian_part_1 [d, p, p]:", hessian_part_1.shape)

        hessian_part_2 = (psi_order_1 / psi_order_0)[:, None] * (
            psi_order_1 / psi_order_0
        )[:, :, None]
        # print("hessian_part_2 [d, p, p]:", hessian_part_2.shape)

        return hessian_part_1.sum(axis=0) - hessian_part_2.sum(axis=0)


@final
class BreslowPartialLifetimeLikelihood(
    MaximumLikehoodOptimizer[LinearCovarEffect, CoxData]
):
    model: LinearCovarEffect
    data: CoxData
    s_j: NDArray[np.float64]
    scipy_method = "trust-exact"

    def __init__(
        self,
        covar_effect: LinearCovarEffect,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
    ):
        self.model = copy.deepcopy(covar_effect)
        self.data = CoxData(time, covar, event=event, entry=entry)
        self.s_j = np.dot(self.data.death_set, self.data.covar)

    @property
    @override
    def nb_observations(self) -> int:
        return len(self.data.time)

    @override
    def _initialize_model(self) -> LinearCovarEffect:
        self.model.params = np.random.random(self.data.covar.shape[1])
        return self.model

    @override
    def _get_params_bounds(self) -> Bounds:
        return Bounds(
            np.full(self.model.nb_params, -np.inf),
            np.full(self.model.nb_params, np.inf),
        )

    @override
    def negative_log(self, params: Array1D[np.float64]) -> ToFloat:
        self.model.params = params  # changes model params

        return -(
            np.log(self.model.g(self.s_j)).sum()
            - (
                self.data.event_count[:, None] * np.log(psi(self.model, self.data))
            ).sum()
        )

    def jac_negative_log(self, params: Array1D[np.float64]) -> Array1D[np.float64]:
        self.model.params = params  # changes model params

        return -(
            self.s_j.sum(axis=0)
            - (
                self.data.event_count[:, None]
                * (psi(self.model, self.data, order=1) / psi(self.model, self.data))
            ).sum(axis=0)
        )

    def hess_negative_log(self, params: Array1D[np.float64]) -> ToFloat2D:
        self.model.params = params  # changes model params

        psi_order_0 = psi(self.model, self.data)
        psi_order_1 = psi(self.model, self.data, order=1)

        hessian_part_1 = psi(self.model, self.data, order=2) / psi_order_0[:, :, None]
        # print("hessian_part_1 [d, p, p]:", hessian_part_1.shape)

        hessian_part_2 = (psi_order_1 / psi_order_0)[:, None] * (
            psi_order_1 / psi_order_0
        )[:, :, None]
        # print("hessian_part_2 [d, p, p]:", hessian_part_2.shape)

        return (self.data.event_count[:, None, None] * hessian_part_1).sum(axis=0) - (
            self.data.event_count[:, None, None] * hessian_part_2
        ).sum(axis=0)


@final
class EfronPartialLifetimeLikelihood(
    MaximumLikehoodOptimizer[LinearCovarEffect, CoxData]
):
    model: LinearCovarEffect
    data: CoxData
    s_j: NDArray[np.float64]
    discount_rates: NDArray[np.float64]
    discount_rates_mask: NDArray[np.bool_]
    scipy_method = "trust-exact"

    def __init__(
        self,
        covar_effect: LinearCovarEffect,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
    ):
        self.model = copy.deepcopy(covar_effect)
        self.data = CoxData(time, covar, event=event, entry=entry)
        self.s_j = np.dot(self.data.death_set, self.data.covar)
        self.discount_rates = (
            np.vstack(
                (np.arange(self.data.event_count.max()),) * len(self.data.event_count)
            )
            / self.data.event_count[:, None]
        )
        self.discount_rates_mask = np.where(self.discount_rates < 1, True, False)

    @property
    @override
    def nb_observations(self) -> int:
        return len(self.data.time)

    @override
    def _initialize_model(self) -> LinearCovarEffect:
        self.model.params = np.random.random(self.data.covar.shape[1])
        return self.model

    @override
    def _get_params_bounds(self) -> Bounds:
        return Bounds(
            np.full(self.model.nb_params, -np.inf),
            np.full(self.model.nb_params, np.inf),
        )

    def _psi_efron(
        self,
        order: Literal[0] | Literal[1] | Literal[2] = 0,
    ) -> NDArray[np.float64]:
        """Psi formula for Efron method

        Args:
            order (int, optional): order derivatives with respect to params. Defaults to 0.

        Returns:
            np.ndarray: psi formulation for Efron method
            If order 0, shape [m, max(d_j)]
            If order 1, shape [m, max(d_j), p]
            If order 2, shape [m, max(d_j), p, p]
        """

        if order == 0:
            # shape [m, max(d_j)]
            return (
                psi(self.model, self.data, order=order) * self.discount_rates_mask
                - psi(self.model, self.data, on="death", order=order)
                * self.discount_rates
                * self.discount_rates_mask
            )
        elif order == 1:
            # shape [m, max(d_j), p]
            return (
                psi(self.model, self.data, order=1)[:, None, :]
                * self.discount_rates_mask[:, :, None]
                - psi(self.model, self.data, on="death", order=1)[:, None, :]
                * (self.discount_rates * self.discount_rates_mask)[:, :, None]
            )
        elif order == 2:
            # shape [m, max(d_j), p, p]
            return (
                psi(self.model, self.data, order=2)[:, None, :]
                * self.discount_rates_mask[:, :, None, None]
                - psi(self.model, self.data, on="death", order=2)[:, None, :]
                * (self.discount_rates * self.discount_rates_mask)[:, :, None, None]
            )

    @override
    def negative_log(
        self,
        params: NDArray[np.float64],
    ) -> float:
        self.model.params = params  # changes model params

        # .sum(axis=1, keepdims=True) --> sum on alpha to d_j
        # .sum() --> sum on j
        # using where in np.log allows to avoid 0. masked elements
        m = self._psi_efron()
        neg_L = -(
            np.log(self.model.g(self.s_j)).sum()
            - np.log(m, out=np.zeros_like(m), where=(m != 0))
            .sum(axis=1, keepdims=True)
            .sum()
        )
        return neg_L

    def jac_negative_log(self, params: Array1D[np.float64]) -> Array1D[np.float64]:
        self.model.params = params  # changes model params
        # .sum(axis=1) --> sum on alpha to d_j
        # .sum(axis=0) --> sum on j
        # using where in np.divide allows to avoid 0. masked elements
        a = self._psi_efron(order=1)
        b = self._psi_efron()[:, :, None]
        return -(
            self.s_j.sum(axis=0)
            - np.divide(a, b, out=np.zeros_like(a), where=(b != 0))
            .sum(axis=1)
            .sum(axis=0)
        )

    def hess_negative_log(self, params: Array1D[np.float64]) -> ToFloat2D:
        self.model.params = params  # changes model params

        psi_order_0 = self._psi_efron()
        psi_order_1 = self._psi_efron(order=1)

        # .sum(axis=1) --> sum on alpha to d_j
        # using where in np.divide allows to avoid 0. masked elements
        a = self._psi_efron(order=2)
        b = psi_order_0[:, :, None, None]
        hessian_part_1 = np.divide(a, b, out=np.zeros_like(a), where=(b != 0)).sum(
            axis=1
        )

        # .sum(axis=1) --> sum on alpha to d_j
        # using where in np.divide allows to avoid 0. masked elements
        b = psi_order_0[:, :, None]
        hessian_part_2 = (
            np.divide(psi_order_1, b, out=np.zeros_like(psi_order_1), where=(b != 0))[
                :, :, None, :
            ]
            * (
                np.divide(
                    psi_order_1, b, out=np.zeros_like(psi_order_1), where=(b != 0)
                )
            )[:, :, :, None]
        )
        hessian_part_2 = hessian_part_2.sum(axis=1)

        return hessian_part_1.sum(axis=0) - hessian_part_2.sum(axis=0)


@dataclass
class SemiParamAFTData:
    time: NDArray[np.float64]
    covar: NDArray[np.float64]
    event: NDArray[np.bool_] | None = None
    entry: NDArray[np.float64] | None = None
    nb_observations: int = field(init=False, repr=False)

    def __post_init__(self):
        self.time = reshape_1d_arg(self.time)
        self.event = (
            reshape_1d_arg(self.event)
            if self.event is not None
            else np.ones_like(self.time, dtype=np.bool_)
        )
        self.entry = (
            reshape_1d_arg(self.entry)
            if self.entry is not None
            else np.zeros_like(self.time, dtype=np.float64)
        )
        sizes = [len(x) for x in (self.time, self.event, self.entry, self.covar)]

        if len(set(sizes)) != 1:
            raise ValueError(
                f""""
                All lifetime data must have the same number of values. Fields
                length are different. Got {tuple(sizes)}.
                """
            )

        self.log_time = np.log(self.time)
        self.log_entry = np.log(self.entry)
        self.nb_observations = self.time.size


@dataclass
class SemiParamAFTFittingResults:
    nb_obversations: int  #: Number of observations (samples)
    optimal_params: NDArray[np.float64] = field(
        repr=False
    )  #: Optimal parameters values
    log_rank_stat: float = field(
        repr=False
    )  #: Log rank statistic value at optimal parameters values


#@njit(float64(float64[:,:], float64[:,:], float64[:,:], float64[:,:]),
      #cache=True, fastmath=False, parallel=False) # fastmath=True improvement is negligible, parallel=True does not work
@njit(fastmath=False, parallel=False)              # Ultimately disappointing, only x2 speed
def _log_rank_stat(
        covar: NDArray[np.float64],
        event: NDArray[np.float64],
        eps_time: NDArray[np.float64],
        eps_entry: NDArray[np.float64]
) -> np.float64:

    N = len(covar)
    S = np.zeros(covar.shape[1], dtype=np.float64)

    for i in prange(N - 1):
        X_diff_ij = covar[i, :] - covar[(i + 1):, :]
        eps_time_sgn_diff_ij = np.sign(eps_time[i] - eps_time[(i + 1):])
        min_eps_time = np.minimum(eps_time[i], eps_time[(i + 1):])
        max_eps_entry = np.maximum(eps_entry[i], eps_entry[(i + 1):])
        oij = (
                event[i] * event[(i + 1):]
                + event[i] * (1 - event[(i + 1):]) * (eps_time[i] < eps_time[(i + 1):])
                + (1 - event[i]) * event[(i + 1):] * (eps_time[i] > eps_time[(i + 1):])
        )
        S -= np.sum(X_diff_ij * eps_time_sgn_diff_ij * (max_eps_entry <= min_eps_time) * oij, axis=0)

    return np.sum(S ** 2) / N ** 2

@njit(fastmath=False, parallel=False)
def _log_rank_stat_smooth(
        covar: NDArray[np.float64],
        event: NDArray[np.float64],
        eps_time: NDArray[np.float64],
        eps_entry: NDArray[np.float64]
) -> np.float64:

    N = len(covar)
    S = np.zeros(covar.shape[1], dtype=np.float64) # gradient de l'objectif de régression rank-based

    for i in prange(N):
        if event[i] == 0:
            continue
        X_diff_ij = covar[i, :] - covar[:, :] # (N,p)
        rij_star = np.sqrt(2 / N * np.sum(X_diff_ij ** 2, axis=1)) # (N,)
        eps_time_diff_norm = (eps_time - eps_time[i]) / rij_star # (N,)
        eps_entry_minus_eps_time_norm = (eps_entry - eps_time[i]) / rij_star # (N,)
        S += np.sum(X_diff_ij * (erf(eps_time_diff_norm) - erf(eps_entry_minus_eps_time_norm)), axis=0) # (p,)

    return np.sum(S ** 2) / N ** 2  # On nullifie les composantes du gradient par minimisation scalaire de sa norme L2

@njit(fastmath=False, parallel=False)
def _jac_log_rank_stat_smooth(
        covar: NDArray[np.float64],
        event: NDArray[np.float64],
        eps_time: NDArray[np.float64],
        eps_entry: NDArray[np.float64]
) -> np.float64:

    N = len(covar)
    S = np.zeros((covar.shape[1], covar.shape[1]), dtype=np.float64) # jacobienne du gradient de l'objectif de régression rank-based

    for i in prange(N):
        if event[i] == 0:
            continue
        X_diff_ij = covar[i, :] - covar[:, :] # (N,p)
        rij_star = np.sqrt(2 / N * np.sum(X_diff_ij ** 2, axis=1)) # (N,)
        eps_time_diff_norm = (eps_time - eps_time[i]) / rij_star # (N,)
        eps_entry_minus_eps_time_norm = (eps_entry - eps_time[i]) / rij_star # (N,)
        X_diff_mult_outer = np.multiply.outer(X_diff_ij, X_diff_ij) # (N,p,N,p)
        X_diff_mult_outer_reshaped = np.transpose(np.diagonal(X_diff_mult_outer, axis1=0, axis2=2), axes=(2, 0, 1)) # (N,p,p)
        S += 2 / np.sqrt(np.pi) * np.sum(
            X_diff_mult_outer_reshaped
            * (np.exp(eps_time_diff_norm) - np.exp(eps_entry_minus_eps_time_norm))
            / rij_star, axis=0) # (p,p)

    rank_stat = _log_rank_stat_smooth(covar, event, eps_time, eps_entry)

    return 2 / N ** 2 * S @ rank_stat # (p,), car S est symétrique (car en réalité une hessienne), pas besoin de transposer


class SemiParametricAcceleratedFailureTime:
    """
    Class for semi-parametric, accelerated failure time, model

    Implementation comes from:
    "Semiparametric inference for an accelerated failure time model with dependent truncation";
    Emura & Wang; Ann. Inst. Stat. Math. 2016

    Implementation assumes independence between asset lifetime and both truncation and censoring times
    """

    fitting_results: SemiParamAFTFittingResults | None
    covar_effect: LinearCovarEffect | None
    _training_data: SemiParamAFTData | None
    _sf: NDArray[np.void] | None

    def __init__(self):
        self._sf = None
        self._training_data = None
        self.fitting_results = None
        self.covar_effect = None

    @staticmethod
    def update_params(fun: Callable) -> Callable:
        def wrapper(self, params: NDArray[np.float64], *args, **kwargs):
            self.covar_effect.params = params
            return fun(self, params, *args, **kwargs)
        return wrapper

    @property
    def params(self):
        if self.covar_effect is None:
            return None
        return self.covar_effect.params

    @property
    def nb_params(self):
        if self.covar_effect is None:
            return None
        return self.covar_effect.nb_params

    def _log_time_residuals(self) -> NDArray[np.float64]:
        if self._training_data is None:
            raise ValueError("You need data to compute log_residuals")
        return self._training_data.log_time - self.covar_effect.g(self._training_data.covar, log_scale=True)

    def _log_entry_residuals(self) -> NDArray[np.float64]:
        if self._training_data is None:
            raise ValueError("You need data to compute log_residuals")
        return self._training_data.log_entry - self.covar_effect.g(self._training_data.covar, log_scale=True)

    @update_params
    def log_rank_stat(self, params: NDArray[np.float64]) -> np.float64:
        eps_time = self._log_time_residuals()
        eps_entry = self._log_entry_residuals()
        covar = self._training_data.covar
        event = self._training_data.event

        return _log_rank_stat(covar, event, eps_time, eps_entry)

    def fit(
            self,
            time: NDArray[np.float64],
            covar: NDArray[np.float64],
            event: NDArray[np.bool_] | None = None,
            entry: NDArray[np.float64] | None = None,
            **kwargs
    ) -> SemiParamAFTFittingResults:
        # Init covar_effect
        self.covar_effect = LinearCovarEffect(
            (None,) * np.atleast_2d(np.asarray(covar, dtype=np.float64)).shape[-1]
        )

        # Build training_data
        self._training_data = SemiParamAFTData(
            time=time, covar=covar, event=event, entry=entry
        )

        # Set optimizer and minimize
        x0 = kwargs.pop("x0", np.zeros(covar.shape[1], dtype=np.float64))
        method = kwargs.pop("method", "Nelder-Mead")
        bounds = kwargs.pop("bounds", None)

        optimizer = minimize(
            self.log_rank_stat,
            x0=x0,
            method=method,
            bounds=bounds,
            **kwargs
        )

        # Set output
        optimal_params = np.copy(optimizer.x)
        self.covar_effect.params = optimal_params

        return SemiParamAFTFittingResults(
            nb_obversations=self._training_data.nb_observations,
            optimal_params=optimal_params,
            log_rank_stat=optimizer.fun
        )


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    # Données
    channing_data = pd.read_csv(Path(r"D:\Projets\RTE\ReLife") / "channing.csv", sep=";", decimal=",")
    channing_data = (
        channing_data
        .drop(columns="time")
        .rename(columns={"exit": "time", "cens": "event"})
    )
    time, event, entry = channing_data["time"].values, channing_data["event"].astype(float).values, channing_data[
        "entry"].values
    covar = (channing_data[["sex"]] == "Male").astype(float).values

    # Test fit
    model = SemiParametricAcceleratedFailureTime()
    model.fit(
        time=time, covar=covar, event=event, entry=entry, bracket=(-0.07, 0.07)
    )
    print(model.params)






