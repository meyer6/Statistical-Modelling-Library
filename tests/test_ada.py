import copy
import numpy as np
import pytest

# Replace `your_module` with the actual import path
from stats_library.ensemble import AdaBoostClassifier

class StubEstimator:
    """A base stub that records fit calls and returns a fixed prediction pattern."""
    def __init__(self, predict_pattern, **kwargs):
        # predict_pattern: array-like of length n_samples to be returned by predict
        self.predict_pattern = np.array(predict_pattern)
        self.fit_called_with = None

    def fit(self, X, y, sample_weight=None):
        # Record the sample_weight passed in, for testing
        self.fit_called_with = np.array(sample_weight, copy=True)
        return self

    def predict(self, X):
        # Repeat or truncate the pattern to match X.shape[0]
        n = X.shape[0]
        return np.resize(self.predict_pattern, n)


def test_predict_before_fit_raises():
    clf = AdaBoostClassifier(base_estimator=StubEstimator([1, -1]), n_estimators=5)
    X = np.zeros((4, 2))
    with pytest.raises(ValueError):
        clf.predict(X)


def test_all_correct_yields_full_rounds_and_constant_weights():
    # Stub always predicts y correctly
    y = np.array([1, -1, 1, -1])
    X = np.zeros((4, 2))
    stub = StubEstimator(predict_pattern=y)
    clf = AdaBoostClassifier(base_estimator=stub, n_estimators=3, learning_rate=0.7)
    clf.fit(X, y)

    # Since error=0 each round, it never breaks early
    assert len(clf.estimators_) == 3
    assert len(clf.estimator_weights_) == 3

    # All weights should be equal (because error constant)
    w0 = clf.estimator_weights_[0]
    assert all(np.isclose(w, w0) for w in clf.estimator_weights_)

    # And each stub received a sample_weight vector summing to 1
    for est in clf.estimators_:
        sw = est.fit_called_with
        assert sw is not None
        assert np.isclose(sw.sum(), 1.0)
        assert sw.shape == (4,)


def test_all_wrong_triggers_immediate_break_and_no_estimators():
    # Stub always predicts the opposite of y (so error=1.0)
    y = np.array([1, 1, -1, -1])
    X = np.zeros((4, 2))
    stub = StubEstimator(predict_pattern=-y)
    clf = AdaBoostClassifier(base_estimator=stub, n_estimators=5, learning_rate=1.0)
    clf.fit(X, y)

    # error = 1 ≥ 0.5, so break before any appends
    assert clf.estimators_ == []
    assert clf.estimator_weights_ == []

    # And predict still raises
    with pytest.raises(ValueError):
        clf.predict(X)


# def test_sample_weight_passed_into_estimator_fit():
#     # Use a stub that flips only the first sample wrong
#     y = np.array([1, -1,  1, -1, 1])
#     X = np.zeros((5, 1))
#     # first round: stub misclassifies only index 0
#     pattern = np.array([-1, -1, 1, -1, 1])
#     stub = StubEstimator(predict_pattern=pattern)
#     clf = AdaBoostClassifier(base_estimator=stub, n_estimators=1, learning_rate=0.5)
#     clf.fit(X, y)

#     # Check that the stub's fit was called with the initial uniform weights
#     sw0 = stub.fit_called_with
#     assert sw0.shape == (5,)
#     assert np.allclose(sw0, np.ones(5) / 5)


def test_predict_combination_of_estimators():
    # Manually inject two estimators with known behavior
    X = np.zeros((6, 2))
    # e1: +1 everywhere; e2: -1 on first three, +1 on last three
    e1 = StubEstimator(predict_pattern=[1]*6)
    e2 = StubEstimator(predict_pattern=[-1]*3 + [1]*3)
    clf = AdaBoostClassifier(n_estimators=2)
    clf.estimators_ = [e1, e2]
    clf.estimator_weights_ = [2.0, 1.0]  # weight e1 twice as much as e2

    preds = clf.predict(X)
    # Weighted sum = 2*e1 + 1*e2 → for first three: 2*(+1) + 1*(-1) = +1 → sign=+1
    #                          last three: 2*(+1) + 1*(+1) = +3 → sign=+1
    assert np.all(preds == 1)


def test_integration_on_linearly_separable_data():
    # A simple 1D dataset perfectly separable by a depth-1 tree
    X = np.array([[0],[0],[1],[1]])
    y = np.array([0,0,1,1])
    clf = AdaBoostClassifier(n_estimators=10, learning_rate=1.0)
    clf.fit(X, y)
    preds = clf.predict(X)
    print(preds)
    # Should classify perfectly
    assert np.array_equal(preds, y)