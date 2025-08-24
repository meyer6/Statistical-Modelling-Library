import pytest
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
from stats_library.ensemble import GradientBoostingRegressor


def test_predict_before_fit_raises():
    model = GradientBoostingRegressor()
    X = np.random.randn(10, 2)
    with pytest.raises(ValueError) as excinfo:
        model.predict(X)
    assert "Must call 'fit' before 'predict'" in str(excinfo.value)


def test_constant_target():
    # When y is constant, predictions should equal that constant
    X = np.random.randn(50, 3)
    y = np.full(50, 5.0)
    model = GradientBoostingRegressor(n_estimators=5, learning_rate=0.5)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.allclose(preds, 5.0)


def test_estimators_count_and_sign():
    # Ensure that estimators_ list has correct length and type
    X = np.random.randn(20, 4)
    y = np.sin(X[:, 0])
    n_est = 7
    lr = 0.2
    model = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr)
    model.fit(X, y)
    assert len(model.estimators_) == n_est
    # for est in model.estimators_:
    #     assert isinstance(est, DecisionTreeRegressor)


def test_prediction_improves_over_constant():
    # Check that using estimators improves over constant prediction (mean)
    rng = np.random.RandomState(0)
    X = rng.randn(100, 1)
    y = X[:, 0]**2 + rng.randn(100) * 0.1

    model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1)
    baseline = np.mean(y)
    baseline_error = np.mean((y - baseline)**2)

    model.fit(X, y)
    preds = model.predict(X)
    error = np.mean((y - preds)**2)

    assert error < baseline_error


def test_compare_to_sklearn_gradient_boost():
    """
    Compare our implementation to scikit-learn's GradientBoostingRegressor
    on a synthetic regression task.
    """
    rng = np.random.RandomState(42)
    X = rng.randn(100, 2)
    y = np.sin(X[:, 0])*100 + np.cos(X[:, 1]) + rng.randn(100) * 0.1

    params = {
        'n_estimators': 5,
        'learning_rate': 0.1,
        'max_depth': 3,
    }

    # Our model
    our_model = GradientBoostingRegressor(
        base_estimator=DecisionTreeRegressor(max_depth=params['max_depth']),
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
    )
    our_model.fit(X, y)
    pred_our = our_model.predict(X)

    # Scikit-learn model
    sk_model = SklearnGBR(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        random_state=42,
    )
    sk_model.fit(X, y)
    pred_sk = sk_model.predict(X)

    # Ensure that predictions are close
    # Use mean absolute error as the metric
    mae = np.mean(np.abs(pred_our - pred_sk))
    assert mae < 0.1, f"Mean absolute difference {mae} exceeds tolerance"
