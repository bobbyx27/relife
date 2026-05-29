import numpy as np
import pytest

from relife.lifetime_model import (
    Gamma,
    Mixture,
    ProportionalHazard,
    Weibull,
)
from relife.lifetime_model.distribution import LifetimeDistribution
from relife.lifetime_model.regression import LifetimeRegression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_weibull():
    """Pre-fitted 2-component Weibull mixture (distribution family)."""
    m = Mixture(Weibull(), Weibull())
    m.components[0].params = np.array([2.0, 100.0])
    m.components[1].params = np.array([5.0, 300.0])
    m._weights = np.array([0.4, 0.6])
    return m


@pytest.fixture
def rc_data(two_weibull):
    """Right-censored sample from the two_weibull fixture."""
    rng = np.random.default_rng(0)
    n = 400
    time = two_weibull.rvs(n, seed=rng)
    event = rng.random(n) > 0.15
    return time, event


@pytest.fixture
def ltrc_data(two_weibull):
    """Left-truncated right-censored sample from two_weibull."""
    rng = np.random.default_rng(1)
    n = 400
    time_full = two_weibull.rvs(n, seed=rng)
    entry = rng.uniform(0, time_full * 0.3)
    event = rng.random(n) > 0.2
    mask = time_full > entry
    return time_full[mask], event[mask], entry[mask]


# ---------------------------------------------------------------------------
# 1 — Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_two_components_default_weights(self):
        m = Mixture(Weibull(), Weibull())
        assert m.nb_components == 2
        np.testing.assert_allclose(m.weights, [0.5, 0.5])

    def test_three_components_custom_weights(self):
        m = Mixture(Weibull(), Gamma(), Weibull(), weights=np.array([0.2, 0.5, 0.3]))
        assert m.nb_components == 3
        np.testing.assert_allclose(m.weights.sum(), 1.0)

    def test_too_few_components_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            Mixture(Weibull())

    def test_mixed_families_raises(self):
        with pytest.raises(TypeError, match="same model family"):
            Mixture(Weibull(), ProportionalHazard(Weibull()))

    def test_weights_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            Mixture(Weibull(), Weibull(), weights=np.array([0.5, 0.3, 0.2]))

    def test_weights_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1"):
            Mixture(Weibull(), Weibull(), weights=np.array([0.3, 0.3]))

    def test_nonpositive_weights_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            Mixture(Weibull(), Weibull(), weights=np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# 2 — Survival functions
# ---------------------------------------------------------------------------


class TestSurvivalFunctions:
    def test_sf_matches_weighted_sum(self, two_weibull):
        t = np.array([50.0, 100.0, 200.0, 400.0])
        c0, c1 = two_weibull.components
        w = two_weibull.weights
        expected = w[0] * c0.sf(t) + w[1] * c1.sf(t)
        np.testing.assert_allclose(two_weibull.sf(t), expected)

    def test_pdf_matches_weighted_sum(self, two_weibull):
        t = np.array([50.0, 100.0, 200.0, 400.0])
        c0, c1 = two_weibull.components
        w = two_weibull.weights
        expected = w[0] * c0.pdf(t) + w[1] * c1.pdf(t)
        np.testing.assert_allclose(two_weibull.pdf(t), expected)

    def test_hf_equals_pdf_over_sf(self, two_weibull):
        t = np.linspace(10, 500, 50)
        np.testing.assert_allclose(
            two_weibull.hf(t),
            two_weibull.pdf(t) / two_weibull.sf(t),
            rtol=1e-10,
        )

    def test_chf_equals_neg_log_sf(self, two_weibull):
        t = np.linspace(10, 500, 50)
        np.testing.assert_allclose(
            two_weibull.chf(t),
            -np.log(two_weibull.sf(t)),
            rtol=1e-10,
        )

    def test_sf_boundary(self, two_weibull):
        assert float(np.ravel(two_weibull.sf(np.array([0.0])))[0]) == pytest.approx(1.0, abs=1e-6)

    def test_sf_decreasing(self, two_weibull):
        t = np.linspace(1, 1000, 200)
        sf_vals = two_weibull.sf(t)
        assert np.all(np.diff(sf_vals) <= 0)


# ---------------------------------------------------------------------------
# 3 — Random variate sampling
# ---------------------------------------------------------------------------


class TestRvs:
    def test_shape_1d(self, two_weibull):
        samples = two_weibull.rvs(200, seed=42)
        assert samples.shape == (200,)

    def test_shape_2d(self, two_weibull):
        samples = two_weibull.rvs(50, nb_assets=10, seed=42)
        assert samples.shape == (10, 50)

    def test_nonnegative(self, two_weibull):
        samples = two_weibull.rvs(500, seed=42)
        assert np.all(samples >= 0)

    def test_return_event_entry(self, two_weibull):
        result = two_weibull.rvs(100, return_event=True, return_entry=True, seed=42)
        assert len(result) == 3
        times, events, entries = result
        assert times.shape == (100,)
        assert events.dtype == np.bool_
        assert np.all(entries == 0.0)


# ---------------------------------------------------------------------------
# 4 — Fit on right-censored data
# ---------------------------------------------------------------------------


class TestFitRC:
    def test_weights_sum_to_one(self, rc_data):
        time, event = rc_data
        m = Mixture(Weibull(), Weibull())
        m.fit(time, event=event)
        assert float(m.weights.sum()) == pytest.approx(1.0, abs=1e-10)

    def test_weights_positive(self, rc_data):
        time, event = rc_data
        m = Mixture(Weibull(), Weibull())
        m.fit(time, event=event)
        assert np.all(m.weights > 0)

    def test_params_finite(self, rc_data):
        time, event = rc_data
        m = Mixture(Weibull(), Weibull())
        m.fit(time, event=event)
        assert np.all(np.isfinite(m.params))
        assert np.all(np.isfinite(m.weights))

    def test_log_likelihood_finite(self, rc_data):
        time, event = rc_data
        m = Mixture(Weibull(), Weibull())
        m.fit(time, event=event)
        assert np.isfinite(m.fitting_results.neg_log_likelihood)


# ---------------------------------------------------------------------------
# 5 — Fit on LTRC data
# ---------------------------------------------------------------------------


class TestFitLTRC:
    def test_converges(self, ltrc_data):
        time, event, entry = ltrc_data
        m = Mixture(Weibull(), Weibull())
        m.fit(time, event=event, entry=entry)
        assert np.all(np.isfinite(m.weights))
        assert np.all(np.isfinite(m.params))

    def test_weights_sum_to_one(self, ltrc_data):
        time, event, entry = ltrc_data
        m = Mixture(Weibull(), Weibull())
        m.fit(time, event=event, entry=entry)
        assert float(m.weights.sum()) == pytest.approx(1.0, abs=1e-10)

    def test_ll_better_than_single_component(self, ltrc_data):
        time, event, entry = ltrc_data
        m2 = Mixture(Weibull(), Weibull())
        m2.fit(time, event=event, entry=entry)
        single = Weibull()
        single.fit(time, event=event, entry=entry)
        # Mixture should have a lower or equal neg-log-likelihood (better fit)
        assert m2.fitting_results.neg_log_likelihood <= single.fitting_results.neg_log_likelihood + 1.0


# ---------------------------------------------------------------------------
# 6 — EM monotonicity
# ---------------------------------------------------------------------------


class TestEMMonotonicity:
    def test_log_likelihood_nondecreasing(self, rc_data):
        time, event = rc_data
        ll_history = []

        class _TrackedMixture(Mixture):
            def _log_likelihood(self, t, *args, event, entry):
                ll = super()._log_likelihood(t, *args, event=event, entry=entry)
                ll_history.append(ll)
                return ll

        m = _TrackedMixture(Weibull(), Weibull())
        m.fit(time, event=event)

        assert len(ll_history) >= 2
        diffs = np.diff(ll_history)
        assert np.all(diffs >= -1e-5), f"EM not monotone, min drop: {diffs.min():.2e}"


# ---------------------------------------------------------------------------
# 7 — FittingResults
# ---------------------------------------------------------------------------


class TestFittingResults:
    def test_not_none_after_fit(self, rc_data):
        time, event = rc_data
        m = Mixture(Weibull(), Weibull())
        m.fit(time, event=event)
        assert m.fitting_results is not None

    def test_optimal_params_length(self, rc_data):
        time, event = rc_data
        K = 2
        m = Mixture(Weibull(), Weibull())
        m.fit(time, event=event)
        n_component_params = sum(len(c.params) for c in m.components)
        expected_len = (K - 1) + n_component_params
        assert len(m.fitting_results.optimal_params) == expected_len

    def test_aic_bic_finite(self, rc_data):
        time, event = rc_data
        m = Mixture(Weibull(), Weibull())
        m.fit(time, event=event)
        assert np.isfinite(m.fitting_results.AIC)
        assert np.isfinite(m.fitting_results.BIC)


# ---------------------------------------------------------------------------
# 8 — Regression components
# ---------------------------------------------------------------------------


class TestFitRegression:
    def test_regression_mixture_fits(self):
        rng = np.random.default_rng(7)
        n = 300
        # 1-column covariate — PH(Weibull()) defaults to 3 params (shape, scale, beta)
        covar = rng.standard_normal((n, 1))

        ph0 = ProportionalHazard(Weibull())
        ph0.params = np.array([2.0, 100.0, 0.5])
        ph1 = ProportionalHazard(Weibull())
        ph1.params = np.array([5.0, 300.0, -0.3])
        true_mix = Mixture(ph0, ph1)
        true_mix._weights = np.array([0.4, 0.6])

        time = true_mix.rvs(n, covar, seed=rng)
        event = rng.random(n) > 0.15

        m = Mixture(ProportionalHazard(Weibull()), ProportionalHazard(Weibull()))
        m.fit(time, covar, event=event)

        assert np.all(np.isfinite(m.weights))
        assert np.all(np.isfinite(m.params))
        assert float(m.weights.sum()) == pytest.approx(1.0, abs=1e-10)
        assert m.fitting_results is not None
