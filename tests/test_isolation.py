import numpy as np
import pytest
from stats_library.tree._isolation import IsolationTree
from stats_library.data_preparation.anomolies._iforest import IsolationForest

@ pytest.fixture(autouse=True)
def seed_rng(monkeypatch):
    # Ensure reproducibility for random_state None
    np.random.seed(42)
    yield


def test_fit_creates_estimators():
    X = np.random.randn(100, 3)
    forest = IsolationForest(n_estimators=5, max_samples=20, random_state=0)
    forest.fit(X)
    assert len(forest.estimators_) == 5
    for tree in forest.estimators_:
        assert isinstance(tree, IsolationTree)


def test_detects_obvious_outlier():
    # Single large outlier among normals
    X = np.vstack([np.random.randn(99, 2), np.array([10, 10])])
    forest = IsolationForest(n_estimators=50, max_samples='auto', contamination=0.01, random_state=42)
    forest.fit(X)
    preds = forest.predict(X)
    # The last point should be labeled as anomaly
    assert preds[-1] == -1


def test_max_samples_auto_setting():
    X = np.random.randn(100, 2)
    forest = IsolationForest(n_estimators=5, max_samples='auto')
    forest.fit(X)
    # max_samples should be min(256, n_samples)
    assert forest.max_samples == 100


def test_predict_without_fit_raises():
    X = np.random.randn(10, 3)
    forest = IsolationForest(n_estimators=5)
    with pytest.raises(ValueError):
        forest.predict(X)
