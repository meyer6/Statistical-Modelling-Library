import numpy as np
import pytest
from stats_library.tree._decision import DecisionTreeClassifier, DecisionTreeRegressor

# Helper to compare arrays

def assert_array_almost_equal(a, b, tol=1e-6):
    assert np.allclose(a, b, atol=tol), f"Arrays not almost equal: {a} vs {b}"

class TestDecisionTreeClassifier:
    def test_simple_perfect_split(self):
        # Perfectly separable: one feature
        X = np.array([[0], [1], [0], [1]])
        y = np.array([0, 1, 0, 1])
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        print(clf.predict_proba(X), "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
        assert np.array_equal(preds, y)

    def test_predict_proba_uniform_weights(self):
        # Without sample_weight, uniform probabilities at root
        X = np.array([[0], [0], [1], [1]])
        y = np.array([0, 0, 1, 1])
        clf = DecisionTreeClassifier(max_depth=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        # With no splitting (max_depth=0), leaf sees equal counts of classes
        expected = np.array([[0.5, 0.5]] * 4)
        assert_array_almost_equal(proba, expected)

    def test_sample_weight_affects_proba(self):
        # Two samples, different weights
        X = np.array([[0], [1]])
        y = np.array([0, 1])
        weights = np.array([0.9, 0.1])
        clf = DecisionTreeClassifier(max_depth=0)
        clf.fit(X, y, sample_weight=weights)
        proba = clf.predict_proba(X)
        # Single leaf, class0 weight 0.9, class1 weight 0.1
        expected = np.array([[0.9, 0.1], [0.9, 0.1]])
        assert_array_almost_equal(proba, expected)

    def test_min_samples_leaf(self):
        # Ensure min_samples_leaf prevents split
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 0, 1, 1])
        clf = DecisionTreeClassifier(min_samples_leaf=2, max_depth=2)
        clf.fit(X, y)
        # With min_samples_leaf=2, root split on threshold=1.5 yields left size=2, right size=2 allowed,
        # but deeper splits prevented by leaf count
        preds = clf.predict(X)
        assert_array_almost_equal(np.unique(preds), np.array([0, 1]))

class TestDecisionTreeRegressor:
    def test_simple_regression(self):
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        reg = DecisionTreeRegressor()
        reg.fit(X, y)
        preds = reg.predict(X)
        # Perfect split at threshold=1.5
        assert_array_almost_equal(preds, y)

    def test_leaf_mean_uniform_weights(self):
        # No split (max_depth=0), leaf should predict overall mean
        X = np.array([[0], [1], [2]])
        y = np.array([0.0, 1.0, 2.0])
        reg = DecisionTreeRegressor(max_depth=0)
        reg.fit(X, y)
        preds = reg.predict(X)
        expected = np.array([1.0, 1.0, 1.0])
        assert_array_almost_equal(preds, expected)

    def test_sample_weight_in_regression(self):
        # Weighted average at leaf
        X = np.array([[0], [1], [2]])
        y = np.array([0.0, 10.0, 20.0])
        weights = np.array([0.1, 0.1, 0.8])
        reg = DecisionTreeRegressor(max_depth=0)
        reg.fit(X, y, sample_weight=weights)
        preds = reg.predict(X)
        # weighted mean = (0*0.1 + 10*0.1 + 20*0.8) / 1.0 = 16.0
        expected = np.array([17.0, 17.0, 17.0])
        assert_array_almost_equal(preds, expected)

    def test_min_samples_split_prevents_splitting(self):
        # min_samples_split > n_samples prevents any split
        X = np.array([[0], [1], [2]])
        y = np.array([0.0, 1.0, 2.0])
        reg = DecisionTreeRegressor(min_samples_split=4)
        reg.fit(X, y)
        preds = reg.predict(X)
        # single leaf mean = 1.0
        expected = np.array([1.0, 1.0, 1.0])
        assert_array_almost_equal(preds, expected)

if __name__ == "__main__":
    pytest.main()