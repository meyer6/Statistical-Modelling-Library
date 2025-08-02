import numpy as np
import pytest
from stats_library.tree._decision import DecisionTreeRegressor, DecisionNode


def test_constant_target():
    # If y is constant, tree should predict that constant value everywhere
    X = np.random.rand(10, 3)
    y = np.full(10, 5.0)
    tree = DecisionTreeRegressor()
    tree.fit(X, y)
    preds = tree.predict(X)
    assert np.all(preds == 5.0)


def test_simple_split_mean():
    # Simple one-feature split: two groups with different means
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 10, 10])
    tree = DecisionTreeRegressor(max_depth=1)
    tree.fit(X, y)
    # Check root is a split node
    root = tree._tree
    assert not root.is_leaf_node()
    # Predictions: below threshold ~0, above ~10
    preds = tree.predict(X)
    assert np.allclose(preds[:2], 0, atol=1e-6)
    assert np.allclose(preds[2:], 10, atol=1e-6)


def test_max_depth_limit():
    # Depth limit should force a leaf at root
    X = np.random.rand(50, 2)
    y = np.linspace(0, 1, 50)
    tree = DecisionTreeRegressor(max_depth=0)
    tree.fit(X, y)
    # Root should be a leaf
    assert tree._tree.is_leaf_node()
    # Predict should be mean of y
    assert np.allclose(tree.predict(X), np.mean(y))


def test_min_samples_split():
    # If less samples than min_samples_split, tree remains a single leaf
    X = np.random.rand(3, 2)
    y = np.array([1, 2, 3])
    tree = DecisionTreeRegressor(min_samples_split=5)
    tree.fit(X, y)
    assert tree._tree.is_leaf_node()
    assert np.allclose(tree.predict(X), np.mean(y))


def test_min_samples_leaf():
    # Ensure leaf constraint: no group smaller than min_samples_leaf
    X = np.array([[0], [0], [1], [1], [1]])
    y = np.array([0, 0, 1, 1, 1])
    tree = DecisionTreeRegressor(min_samples_leaf=2)
    tree.fit(X, y)
    # After split, one side would have only one sample => should avoid split
    assert tree._tree.is_leaf_node()


def test_invalid_max_features():
    with pytest.raises(ValueError):
        DecisionTreeRegressor(max_features='invalid')._get_candidate_features(4)


def test_max_features_int_and_float_sqrt_log2():
    # Test that valid settings return correct number of features
    n_features = 10
    # int
    sel = DecisionTreeRegressor(max_features=3)._get_candidate_features(n_features)
    assert len(sel) == 3
    # float
    sel = DecisionTreeRegressor(max_features=0.5)._get_candidate_features(n_features)
    assert len(sel) == max(1, int(0.5 * n_features))
    # sqrt
    sel = DecisionTreeRegressor(max_features='sqrt')._get_candidate_features(n_features)
    assert len(sel) == max(1, int(np.sqrt(n_features)))
    # log2
    sel = DecisionTreeRegressor(max_features='log2')._get_candidate_features(n_features)
    assert len(sel) == max(1, int(np.log2(n_features)))


def test_variance_reduction():
    # Simple variance reduction test
    y = np.array([0, 0, 10, 10])
    left_y = np.array([0, 0])
    right_y = np.array([10, 10])
    vr = DecisionTreeRegressor()._variance_reduction(y, left_y, right_y)
    # Total variance = 25, after perfect split children variance=0 => reduction=25
    assert pytest.approx(vr) == np.var(y)


def test_predict_single_sample():
    # Ensure predict works for single sample array
    X = np.array([[0], [2]])
    y = np.array([0, 4])
    tree = DecisionTreeRegressor()
    tree.fit(X, y)
    pred = tree.predict(np.array([[1]]))
    assert pred.shape == (1,)

# End of test suite
