import pytest
from stats_library.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor as RF

from stats_library.metrics._regressor import mean_absolute_error

import numpy as np
import pytest



def make_classification_data(n_samples=100, n_features=5, n_classes=2, random_state=42):
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    # assign labels by sum of features
    y = (X.sum(axis=1) > 0).astype(int)
    return X, y


def make_regression_data(n_samples=100, n_features=5, random_state=42):
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    # target is linear combination + noise
    coef = rng.randn(n_features)
    y = X.dot(coef) + rng.randn(n_samples) * 0.1
    return X, y


def test_classifier_fit_and_predict():
    X, y = make_classification_data()
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    # predictions should be 0 or 1
    assert set(preds) <= {0, 1}
    # accuracy should be high on training
    acc = (preds == y).mean()
    assert acc > 0.9


def test_classifier_predict_proba_shape_and_sum():
    X, y = make_classification_data()
    clf = RandomForestClassifier(n_estimators=5, random_state=1)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], len(np.unique(y)))
    # probabilities sum to 1 for each sample
    sums = proba.sum(axis=1)
    assert np.allclose(sums, 1.0)


def test_regressor_fit_and_predict():
    X, y = make_regression_data()
    # reg = RandomForestRegressor()
    # reg.fit(X, y)
    # preds = reg.predict(X)

    
    reg2 = RF()
    reg2.fit(X, y)
    preds2 = reg2.predict(X)

    # assert preds.shape == (X.shape[0],)
    # mean squared error should be small
    # mse = ((preds - y) ** 2).mean()
    mse2 = ((preds2 - y) ** 2).mean()

    assert mse2 < mse2


def test_random_state_reproducibility_classification():
    X, y = make_classification_data()
    clf1 = RandomForestClassifier(n_estimators=5, random_state=123)
    clf2 = RandomForestClassifier(n_estimators=5, random_state=123)
    clf1.fit(X, y)
    clf2.fit(X, y)
    assert np.array_equal(clf1.predict(X), clf2.predict(X))


def test_random_state_reproducibility_regression():
    X, y = make_regression_data()
    reg1 = RandomForestRegressor(n_estimators=5, random_state=99)
    reg2 = RandomForestRegressor(n_estimators=5, random_state=99)
    reg1.fit(X, y)
    reg2.fit(X, y)
    assert np.allclose(reg1.predict(X), reg2.predict(X))


def test_trees_count_after_fit():
    X, y = make_classification_data()
    n_estimators = 7
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    clf.fit(X, y)
    assert len(clf.trees) == n_estimators


def test_predict_before_fit_raises():
    clf = RandomForestClassifier()
    with pytest.raises(ValueError):
        clf.predict(np.zeros((1, 1)))  # no trees yet

    reg = RandomForestRegressor()
    with pytest.raises(ValueError):
        reg.predict(np.zeros((1, 1)))  # self.trees empty generator sum

