# pyright: basic

import numpy as np
import pytest

from relife.lifetime_models import (
    Gamma,
    MixtureWeightsRegression,
    ParametricLifetimeMixture,
    ParametricLifetimeMixtureWithWeightsRegression,
    ParametricProportionalHazard,
    Weibull,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_weibull():
    """Pre-fitted 2-component Weibull mixture (distribution family)."""
    m = ParametricLifetimeMixture(Weibull(), Weibull())
    m.components[0].set_params(np.array([2.0, 0.01]))
    m.components[1].set_params(np.array([5.0, 1.0 / 300]))
    m._mix_weights = np.array([0.4, 0.6])
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


@pytest.fixture
def rc_data_wr():
    """RC data where true mixing weights depend on a covariate."""
    rng = np.random.default_rng(77)
    n = 250
    nb_coef = 1
    covar = rng.standard_normal((n, nb_coef))
    # True mixing: prob(k=0) = sigmoid(covar[:,0])
    prob_k0 = 1.0 / (1.0 + np.exp(-covar[:, 0]))
    k_assign = (rng.random(n) < prob_k0).astype(int)
    w0 = Weibull(2.0, 0.01)
    w1 = Weibull(5.0, 1.0 / 300)
    time = np.where(
        k_assign == 0,
        np.ravel(w0.rvs(n, seed=rng)),
        np.ravel(w1.rvs(n, seed=rng)),
    )
    event = rng.random(n) > 0.15
    return time, covar, event


@pytest.fixture
def fitted_mixture_wr(rc_data_wr):
    """Fitted ParametricLifetimeMixtureWithWeightsRegression (PPH components, 1 covariate)."""
    time, covar, event = rc_data_wr
    m = ParametricLifetimeMixtureWithWeightsRegression(
        ParametricProportionalHazard(Weibull()),
        ParametricProportionalHazard(Weibull()),
        nb_coef=1,
    )
    m.fit(time, covar, event=event)
    return m, covar


# ---------------------------------------------------------------------------
# 1 — ParametricLifetimeMixture: Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_two_components_default_weights(self):
        m = ParametricLifetimeMixture(Weibull(), Weibull())
        assert m.nb_components == 2
        np.testing.assert_allclose(m.mix_weights, [0.5, 0.5])

    def test_three_components_custom_weights(self):
        m = ParametricLifetimeMixture(Weibull(), Gamma(), Weibull(), mix_weights=np.array([0.2, 0.5, 0.3]))
        assert m.nb_components == 3
        np.testing.assert_allclose(m.mix_weights.sum(), 1.0)

    def test_too_few_components_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            ParametricLifetimeMixture(Weibull())

    def test_mixed_families_raises(self):
        with pytest.raises(TypeError, match="same model family"):
            ParametricLifetimeMixture(Weibull(), ParametricProportionalHazard(Weibull()))

    def test_weights_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            ParametricLifetimeMixture(Weibull(), Weibull(), mix_weights=np.array([0.5, 0.3, 0.2]))

    def test_weights_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1"):
            ParametricLifetimeMixture(Weibull(), Weibull(), mix_weights=np.array([0.3, 0.3]))

    def test_nonpositive_weights_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            ParametricLifetimeMixture(Weibull(), Weibull(), mix_weights=np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# 2 — ParametricLifetimeMixture: Survival functions
# ---------------------------------------------------------------------------


class TestSurvivalFunctions:
    def test_sf_matches_weighted_sum(self, two_weibull):
        t = np.array([50.0, 100.0, 200.0, 400.0])
        c0, c1 = two_weibull.components
        w = two_weibull.mix_weights
        expected = w[0] * c0.sf(t) + w[1] * c1.sf(t)
        np.testing.assert_allclose(two_weibull.sf(t), expected)

    def test_pdf_matches_weighted_sum(self, two_weibull):
        t = np.array([50.0, 100.0, 200.0, 400.0])
        c0, c1 = two_weibull.components
        w = two_weibull.mix_weights
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

    def test_sf_at_zero_is_one(self, two_weibull):
        assert float(np.ravel(two_weibull.sf(np.array([1e-9])))[0]) == pytest.approx(
            1.0, abs=1e-4
        )

    def test_sf_decreasing(self, two_weibull):
        t = np.linspace(1, 1000, 200)
        sf_vals = np.ravel(two_weibull.sf(t))
        assert np.all(np.diff(sf_vals) <= 0)


# ---------------------------------------------------------------------------
# 3 — ParametricLifetimeMixture: Random variate sampling
# ---------------------------------------------------------------------------


class TestRvs:
    def test_shape_1d(self, two_weibull):
        samples = two_weibull.rvs(200, seed=42)
        assert samples.shape == (200,)

    def test_shape_2d(self, two_weibull):
        samples = two_weibull.rvs((10, 50), seed=42)
        assert samples.shape == (10, 50)

    def test_nonnegative(self, two_weibull):
        samples = two_weibull.rvs(500, seed=42)
        assert np.all(samples >= 0)


# ---------------------------------------------------------------------------
# 4 — ParametricLifetimeMixture: Fit on right-censored data
# ---------------------------------------------------------------------------


class TestFitRC:
    def test_weights_sum_to_one(self, rc_data):
        time, event = rc_data
        m = ParametricLifetimeMixture(Weibull(), Weibull())
        m.fit(time, event=event)
        assert float(m.mix_weights.sum()) == pytest.approx(1.0, abs=1e-10)

    def test_weights_positive(self, rc_data):
        time, event = rc_data
        m = ParametricLifetimeMixture(Weibull(), Weibull())
        m.fit(time, event=event)
        assert np.all(m.mix_weights > 0)

    def test_params_finite(self, rc_data):
        time, event = rc_data
        m = ParametricLifetimeMixture(Weibull(), Weibull())
        m.fit(time, event=event)
        assert np.all(np.isfinite(m.get_params()))
        assert np.all(np.isfinite(m.mix_weights))

    def test_log_likelihood_finite(self, rc_data):
        time, event = rc_data
        m = ParametricLifetimeMixture(Weibull(), Weibull())
        m.fit(time, event=event)
        assert np.isfinite(m.fitting_results.neg_log_likelihood)


# ---------------------------------------------------------------------------
# 5 — ParametricLifetimeMixture: Fit on LTRC data
# ---------------------------------------------------------------------------


class TestFitLTRC:
    def test_converges(self, ltrc_data):
        time, event, entry = ltrc_data
        m = ParametricLifetimeMixture(Weibull(), Weibull())
        m.fit(time, event=event, entry=entry)
        assert np.all(np.isfinite(m.mix_weights))
        assert np.all(np.isfinite(m.get_params()))

    def test_weights_sum_to_one(self, ltrc_data):
        time, event, entry = ltrc_data
        m = ParametricLifetimeMixture(Weibull(), Weibull())
        m.fit(time, event=event, entry=entry)
        assert float(m.mix_weights.sum()) == pytest.approx(1.0, abs=1e-10)

    def test_ll_better_than_single_component(self, ltrc_data):
        time, event, entry = ltrc_data
        m2 = ParametricLifetimeMixture(Weibull(), Weibull())
        m2.fit(time, event=event, entry=entry)
        single = Weibull()
        single.fit(time, event=event, entry=entry)
        assert (
            m2.fitting_results.neg_log_likelihood
            <= single.fitting_results.neg_log_likelihood + 1.0
        )


# ---------------------------------------------------------------------------
# 6 — ParametricLifetimeMixture: EM monotonicity
# ---------------------------------------------------------------------------


class TestEMMonotonicity:
    def test_log_likelihood_nondecreasing(self, rc_data):
        time, event = rc_data
        ll_history = []

        class _TrackedMixture(ParametricLifetimeMixture):
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
# 7 — ParametricLifetimeMixture: FittingResults
# ---------------------------------------------------------------------------


class TestFittingResults:
    def test_not_none_after_fit(self, rc_data):
        time, event = rc_data
        m = ParametricLifetimeMixture(Weibull(), Weibull())
        m.fit(time, event=event)
        assert m.fitting_results is not None

    def test_optimal_params_length(self, rc_data):
        time, event = rc_data
        K = 2
        m = ParametricLifetimeMixture(Weibull(), Weibull())
        m.fit(time, event=event)
        n_component_params = sum(c.get_params().size for c in m.components)
        expected_len = (K - 1) + n_component_params
        assert len(m.fitting_results.optimal_params) == expected_len

    def test_aic_bic_finite(self, rc_data):
        time, event = rc_data
        m = ParametricLifetimeMixture(Weibull(), Weibull())
        m.fit(time, event=event)
        assert np.isfinite(m.fitting_results.aic)
        assert np.isfinite(m.fitting_results.bic)

    def test_success_on_convergence(self, rc_data):
        time, event = rc_data
        m = ParametricLifetimeMixture(Weibull(), Weibull())
        m.fit(time, event=event, tol=1e-4)
        assert isinstance(m.fitting_results.success, bool)


# ---------------------------------------------------------------------------
# 8 — ParametricLifetimeMixture: Regression components
# ---------------------------------------------------------------------------


class TestFitRegression:
    def test_regression_mixture_fits(self):
        rng = np.random.default_rng(42)
        n = 300
        covar = rng.standard_normal((n, 1))
        w0 = Weibull(2.0, 0.01)
        w1 = Weibull(5.0, 1.0 / 300)
        k_assign = rng.choice(2, size=n, p=[0.4, 0.6])
        time = np.where(
            k_assign == 0,
            np.ravel(w0.rvs(n, seed=rng)),
            np.ravel(w1.rvs(n, seed=rng)),
        )
        event = rng.random(n) > 0.15

        m = ParametricLifetimeMixture(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
        )
        m.fit(time, covar, event=event)

        assert np.all(np.isfinite(m.mix_weights))
        assert np.all(np.isfinite(m.get_params()))
        assert float(m.mix_weights.sum()) == pytest.approx(1.0, abs=1e-10)
        assert m.fitting_results is not None


# ---------------------------------------------------------------------------
# 9 — MixtureWeightsRegression
# ---------------------------------------------------------------------------


class TestMixtureWeightsRegression:
    NB_COEF = 2
    K = 3
    N = 40

    @pytest.fixture
    def model(self):
        return MixtureWeightsRegression(nb_coef=self.NB_COEF, nb_components=self.K)

    @pytest.fixture
    def covar(self):
        return np.random.default_rng(0).standard_normal((self.N, self.NB_COEF))

    def test_predict_shape(self, model, covar):
        assert model.predict(covar).shape == (self.N, self.K)

    def test_predict_sums_to_one(self, model, covar):
        out = model.predict(covar)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(self.N), atol=1e-6)

    def test_predict_values_in_0_1(self, model, covar):
        out = model.predict(covar)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_call_equals_predict(self, model, covar):
        np.testing.assert_array_equal(model(covar), model.predict(covar))

    def test_params_length(self, model):
        # weight matrix (K × nb_coef) + bias (K,)
        assert len(model.get_params()) == self.K * (self.NB_COEF + 1)

    def test_fit_changes_params(self, model, covar):
        rng = np.random.default_rng(1)
        q = rng.dirichlet(np.ones(self.K), size=self.N).astype(np.float64)
        params_before = model.get_params().copy()
        model.fit(covar, q, max_iter=20)
        assert not np.allclose(model.get_params(), params_before)

    def test_fit_syncs_params_with_torch(self, model, covar):
        import torch
        rng = np.random.default_rng(2)
        q = rng.dirichlet(np.ones(self.K), size=self.N).astype(np.float64)
        model.fit(covar, q, max_iter=20)
        expected = np.concatenate([
            model.torch_module.weight.data.numpy().ravel(),
            model.torch_module.bias.data.numpy().ravel(),
        ])
        np.testing.assert_allclose(model.get_params(), expected, atol=1e-6)

    def test_fit_reduces_loss(self, model, covar):
        """Cross-entropy loss should be lower after fitting than before."""
        import torch
        rng = np.random.default_rng(3)
        q = rng.dirichlet(np.ones(self.K), size=self.N).astype(np.float64)
        loss_fn = torch.nn.CrossEntropyLoss()
        X = torch.tensor(covar.astype(np.float32))
        targets = torch.tensor(q.astype(np.float32))
        with torch.no_grad():
            loss_before = loss_fn(model.torch_module(X), targets).item()
        model.fit(covar, q, max_iter=200)
        with torch.no_grad():
            loss_after = loss_fn(model.torch_module(X), targets).item()
        assert loss_after < loss_before


# ---------------------------------------------------------------------------
# 10 — ParametricLifetimeMixtureWithWeightsRegression: Instantiation
# ---------------------------------------------------------------------------


class TestWRMixtureInstantiation:
    def test_creates_weights_model(self):
        m = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=2,
        )
        assert isinstance(m.weights_model, MixtureWeightsRegression)

    def test_nb_components(self):
        m = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=1,
        )
        assert m.nb_components == 2

    def test_too_few_components_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            ParametricLifetimeMixtureWithWeightsRegression(
                ParametricProportionalHazard(Weibull()), nb_coef=1
            )

    def test_mixed_families_raises(self):
        with pytest.raises(TypeError, match="same model family"):
            ParametricLifetimeMixtureWithWeightsRegression(
                Weibull(), ParametricProportionalHazard(Weibull()), nb_coef=1
            )

    def test_weights_model_params_in_get_params(self):
        nb_coef = 2
        K = 2
        m = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=nb_coef,
        )
        weights_params = len(m.weights_model.get_params())  # K*(nb_coef+1)
        component_params = sum(c.get_params().size for c in m.components)
        assert len(m.get_params()) == weights_params + component_params

    def test_no_mix_weights_property(self):
        """ParametricLifetimeMixtureWithWeightsRegression has no mix_weights attribute."""
        m = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=1,
        )
        assert not hasattr(m, "mix_weights")


# ---------------------------------------------------------------------------
# 11 — ParametricLifetimeMixtureWithWeightsRegression: _get_weights
# ---------------------------------------------------------------------------


class TestWRMixtureGetWeights:
    def test_get_weights_shape(self):
        nb_coef = 2
        n = 10
        m = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=nb_coef,
        )
        covar = np.random.default_rng(0).standard_normal((n, nb_coef))
        w = m._get_weights(covar)
        assert w.shape == (n, m.nb_components)

    def test_get_weights_sums_to_one(self):
        nb_coef = 1
        n = 20
        m = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=nb_coef,
        )
        covar = np.random.default_rng(0).standard_normal((n, nb_coef))
        w = m._get_weights(covar)
        np.testing.assert_allclose(w.sum(axis=1), np.ones(n), atol=1e-6)


# ---------------------------------------------------------------------------
# 12 — ParametricLifetimeMixtureWithWeightsRegression: Survival functions
# ---------------------------------------------------------------------------


class TestWRMixtureSurvivalFunctions:
    def test_sf_matches_formula(self, fitted_mixture_wr):
        m, covar = fitted_mixture_wr
        n = len(covar)
        time = np.linspace(10, 500, n)
        t2d = time.reshape(-1, 1)
        w = m.weights_model.predict(covar)  # (n, K)
        expected = sum(
            w[:, k : k + 1] * comp.sf(t2d, covar)
            for k, comp in enumerate(m.components)
        )
        np.testing.assert_allclose(m.sf(time, covar), expected, rtol=1e-10)

    def test_pdf_matches_formula(self, fitted_mixture_wr):
        m, covar = fitted_mixture_wr
        n = len(covar)
        time = np.linspace(10, 500, n)
        t2d = time.reshape(-1, 1)
        w = m.weights_model.predict(covar)
        expected = sum(
            w[:, k : k + 1] * comp.pdf(t2d, covar)
            for k, comp in enumerate(m.components)
        )
        np.testing.assert_allclose(m.pdf(time, covar), expected, rtol=1e-10)

    def test_hf_equals_pdf_over_sf(self, fitted_mixture_wr):
        m, covar = fitted_mixture_wr
        time = np.linspace(10, 500, len(covar))
        np.testing.assert_allclose(
            m.hf(time, covar),
            m.pdf(time, covar) / m.sf(time, covar),
            rtol=1e-10,
        )

    def test_chf_equals_neg_log_sf(self, fitted_mixture_wr):
        m, covar = fitted_mixture_wr
        time = np.linspace(10, 500, len(covar))
        np.testing.assert_allclose(
            m.chf(time, covar),
            -np.log(m.sf(time, covar)),
            rtol=1e-10,
        )

    def test_sf_output_has_no_broadcasting_artefact(self, fitted_mixture_wr):
        """(n, K) weights must not produce (n, n) output via broadcasting."""
        m, covar = fitted_mixture_wr
        n = len(covar)
        time = np.linspace(10, 500, n)
        result = np.ravel(m.sf(time, covar))
        assert result.shape == (n,)


# ---------------------------------------------------------------------------
# 13 — ParametricLifetimeMixtureWithWeightsRegression: rvs
# ---------------------------------------------------------------------------


class TestWRMixtureRvs:
    def test_shape_1d(self, rc_data_wr):
        time, covar, event = rc_data_wr
        n = len(time)
        m = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=covar.shape[1],
        )
        m.fit(time, covar, event=event)
        samples = m.rvs(n, covar, seed=0)
        assert samples.shape == (n,)

    def test_shape_2d(self, rc_data_wr):
        time, covar, event = rc_data_wr
        m = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=covar.shape[1],
        )
        m.fit(time, covar, event=event)
        m_assets, n_samples = 5, 20
        covar_m = covar[:m_assets]
        samples = m.rvs((m_assets, n_samples), covar_m, seed=0)
        assert samples.shape == (m_assets, n_samples)

    def test_nonnegative(self, rc_data_wr):
        time, covar, event = rc_data_wr
        m = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=covar.shape[1],
        )
        m.fit(time, covar, event=event)
        samples = m.rvs(len(time), covar, seed=0)
        assert np.all(samples >= 0)


# ---------------------------------------------------------------------------
# 14 — ParametricLifetimeMixtureWithWeightsRegression: Fit & FittingResults
# ---------------------------------------------------------------------------


class TestWRMixtureFit:
    def test_fit_sets_fitting_results(self, rc_data_wr):
        time, covar, event = rc_data_wr
        m = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=covar.shape[1],
        )
        m.fit(time, covar, event=event)
        assert m.fitting_results is not None

    def test_fitting_results_params_length(self, rc_data_wr):
        """optimal_params == get_params() since _fitted_mix_params returns []."""
        time, covar, event = rc_data_wr
        m = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=covar.shape[1],
        )
        m.fit(time, covar, event=event)
        assert len(m.fitting_results.optimal_params) == len(m.get_params())

    def test_component_params_finite_after_fit(self, rc_data_wr):
        time, covar, event = rc_data_wr
        m = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=covar.shape[1],
        )
        m.fit(time, covar, event=event)
        assert np.all(np.isfinite(m.get_params()))

    def test_neg_log_likelihood_finite(self, rc_data_wr):
        time, covar, event = rc_data_wr
        m = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=covar.shape[1],
        )
        m.fit(time, covar, event=event)
        assert np.isfinite(m.fitting_results.neg_log_likelihood)

    def test_aic_bic_finite(self, rc_data_wr):
        time, covar, event = rc_data_wr
        m = ParametricLifetimeMixtureWithWeightsRegression(
            ParametricProportionalHazard(Weibull()),
            ParametricProportionalHazard(Weibull()),
            nb_coef=covar.shape[1],
        )
        m.fit(time, covar, event=event)
        assert np.isfinite(m.fitting_results.aic)
        assert np.isfinite(m.fitting_results.bic)
