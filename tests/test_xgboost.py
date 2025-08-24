import pytest
import numpy as np

from stats_library.ensemble._forest import XGBoostRegressor

  # replace 'your_module' with the actual module name

# Helper to compute mean squared error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def test_predict_before_fit_raises():
    model = XGBoostRegressor()
    X = np.random.randn(10, 2)
    with pytest.raises(ValueError):
        model.predict(X)


def test_constant_target():
    X = np.random.randn(50, 3)
    y = np.ones(50) * 5.0
    model = XGBoostRegressor(n_estimators=10, learning_rate=0.5)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.allclose(preds, 5.0, atol=1e-6)


def test_simple_linear_regression():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 1)
    y = 2 * X[:, 0] + 3
    model = XGBoostRegressor(n_estimators=50, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    error = mse(y, preds)
    assert error < 1e-2, f"MSE too high: {error}"


def test_quadratic_function_accuracy():
    # y = x0^2 - 2*x0 + 1
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, size=(200, 1))
    y = X[:, 0]**2 - 2*X[:, 0] + 1
    model = XGBoostRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    error = mse(y, preds)
    # Quadratic is harder; allow higher tolerance
    assert error < 1e-1, f"MSE too high on quadratic: {error}"


def test_zero_learning_rate():
    # Zero learning rate should give constant prediction equal to global mean
    X = np.random.randn(100, 2)
    y = np.random.randn(100)
    model = XGBoostRegressor(n_estimators=50, learning_rate=0.0)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.allclose(preds, np.mean(y), atol=1e-8)


def test_increasing_estimators_reduces_error():
    rng = np.random.RandomState(1)
    X = rng.randn(200, 2)
    y = 4 * X[:, 0] - 2 * X[:, 1] + 1
    errors = []
    for n in [1, 5, 20, 50]:
        model = XGBoostRegressor(n_estimators=n, learning_rate=0.2)
        model.fit(X, y)
        preds = model.predict(X)
        errors.append(mse(y, preds))
    assert errors[0] > errors[-1]
    assert all(errors[i] >= errors[i+1] for i in range(len(errors)-1))


def test_poisson_loss_prediction_positive():
    rng = np.random.RandomState(0)
    X = rng.rand(80, 2)
    lam = np.exp(X[:, 0] + 0.5 * X[:, 1])
    y = rng.poisson(lam)
    model = XGBoostRegressor(loss='poisson', n_estimators=30, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds > 0)
    assert preds.shape == y.shape


def test_invalid_loss_raises():
    with pytest.raises(ValueError):
        XGBoostRegressor(loss='invalid')


def test_multi_feature_linear():
    # y = sum of features weighted
    rng = np.random.RandomState(6)
    X = rng.randn(150, 3)
    coeffs = np.array([1.5, -2.0, 0.5])
    y = X.dot(coeffs) + 0.7
    model = XGBoostRegressor()
    model.fit(X, y)
    preds = model.predict(X)
    error = mse(y, preds)
    assert error < 1e-2, f"MSE too high on multi-feature linear: {error}"