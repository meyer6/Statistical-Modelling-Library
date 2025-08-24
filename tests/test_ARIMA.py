# test_time_series.py
import pytest
import numpy as np
from stats_library.time_series._time_series import AR, MA, ARMA


# ---- AR tests ----

def make_ar1_series(phi, intercept, n):
    """Generate a noise‐free AR(1) series of length n."""
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = intercept + phi * y[t-1]
    return y

def test_ar_fit_recovers_parameters():
    phi_true = 1.7
    intercept_true = -0.5
    y = make_ar1_series(phi_true, intercept_true, 50)
    model = AR(p=1)
    model.fit(y)
    # coefficients is length‐1 array
    assert model.coefficients.shape == (1,)
    # should recover exactly in noise‐free case
    np.testing.assert_allclose(model.coefficients[0], phi_true, atol=1e-6)
    assert pytest.approx(model.intercept, abs=1e-6) == intercept_true

def test_ar_predict_one_and_multi_step():
    phi_true = 0.4
    intercept_true = 2.0
    y = make_ar1_series(phi_true, intercept_true, 10)
    model = AR(p=1)
    model.fit(y)
    # one‐step forecast should equal the true next value
    y_hist = y.copy()
    y_true_next = intercept_true + phi_true * y_hist[-1]
    y_pred = model.predict(steps=1)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (1,)
    assert pytest.approx(y_pred[0], abs=1e-6) == y_true_next

    # two‐step: after one‐step, feed in the first pred
    y_pred2 = model.predict(steps=2)
    # manual two‐step: 
    first = y_true_next
    second = intercept_true + phi_true * first
    assert y_pred2.shape == (2,)
    assert pytest.approx(y_pred2[0], abs=1e-6) == first
    assert pytest.approx(y_pred2[1], abs=1e-6) == second

def test_ar_fit_too_short_raises():
    model = AR(p=5)
    with pytest.raises(ValueError):
        model.fit(np.arange(5))   # length == p
