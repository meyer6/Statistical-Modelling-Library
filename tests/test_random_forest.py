import pytest
from stats_library.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor as RF

from stats_library.metrics._regressor import mean_absolute_error

import numpy as np
import pytest

import os
import time
from time import perf_counter

# def make_classification_data(n_samples=100, n_features=5, n_classes=2, random_state=42):
#     rng = np.random.RandomState(random_state)
#     X = rng.randn(n_samples, n_features)
#     # assign labels by sum of features
#     y = (X.sum(axis=1) > 0).astype(int)
#     return X, y


# def make_regression_data(n_samples=100, n_features=5, random_state=42):
#     rng = np.random.RandomState(random_state)
#     X = rng.randn(n_samples, n_features)
#     # target is linear combination + noise
#     coef = rng.randn(n_features)
#     y = X.dot(coef) + rng.randn(n_samples) * 0.1
#     return X, y


# def test_classifier_fit_and_predict():
#     X, y = make_classification_data()
#     clf = RandomForestClassifier(n_estimators=10, random_state=0)
#     clf.fit(X, y)
#     preds = clf.predict(X)
#     # predictions should be 0 or 1
#     assert set(preds) <= {0, 1}
#     # accuracy should be high on training
#     acc = (preds == y).mean()
#     assert acc > 0.9


# def test_classifier_predict_proba_shape_and_sum():
#     X, y = make_classification_data()
#     clf = RandomForestClassifier(n_estimators=5, random_state=1)
#     clf.fit(X, y)
#     proba = clf.predict_proba(X)
#     assert proba.shape == (X.shape[0], len(np.unique(y)))
#     # probabilities sum to 1 for each sample
#     sums = proba.sum(axis=1)
#     assert np.allclose(sums, 1.0)


# # def test_regressor_fit_and_predict():
# #     X, y = make_regression_data()
# #     # reg = RandomForestRegressor()
# #     # reg.fit(X, y)
# #     # preds = reg.predict(X)

    
# #     reg2 = RF()
# #     reg2.fit(X, y)
# #     preds2 = reg2.predict(X)

# #     # assert preds.shape == (X.shape[0],)
# #     # mean squared error should be small
# #     # mse = ((preds - y) ** 2).mean()
# #     mse2 = ((preds2 - y) ** 2).mean()

# #     assert mse2 < mse2


# def test_random_state_reproducibility_classification():
#     X, y = make_classification_data()
#     clf1 = RandomForestClassifier(n_estimators=5, random_state=123)
#     clf2 = RandomForestClassifier(n_estimators=5, random_state=123)

#     clf1.fit(X, y)
#     clf2.fit(X, y)
#     assert np.array_equal(clf1.predict(X), clf2.predict(X))


# def test_random_state_reproducibility_regression():
#     X, y = make_regression_data()
#     reg1 = RandomForestRegressor(n_estimators=5, random_state=99)
#     reg2 = RandomForestRegressor(n_estimators=5, random_state=99)
#     reg1.fit(X, y)
#     reg2.fit(X, y)
#     assert np.allclose(reg1.predict(X), reg2.predict(X))


# def test_trees_count_after_fit():
#     X, y = make_classification_data()
#     n_estimators = 7
#     clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
#     clf.fit(X, y)
#     assert len(clf.trees) == n_estimators


# def test_predict_before_fit_raises():
#     clf = RandomForestClassifier()
#     with pytest.raises(ValueError):
#         clf.predict(np.zeros((1, 1)))  # no trees yet

#     reg = RandomForestRegressor()
#     with pytest.raises(ValueError):
#         reg.predict(np.zeros((1, 1)))  # self.trees empty generator sum



class FakeEstimator:
    """
    Fake estimator that simulates a short blocking operation (sleep) in .fit().
    We make per-tree work small so large n_estimators is feasible. Sleep releases the GIL,
    so ThreadPoolExecutor-based threading will reduce wall-clock time.
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        # ~0.02s per tree by default (4 * 0.005)
        loops = 4
        sleep_each = 0.005
        for _ in range(loops):
            import time
            time.sleep(sleep_each)
        return self


def median(lst):
    s = sorted(lst)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


@pytest.mark.slow
def test_rf_threading_large(monkeypatch):
    """
    Large-ish benchmark comparing single-threaded vs multi-threaded fit wall time.
    Tune via environment variables:

    - RF_N_ESTIMATORS: integer number of trees (default 400)
    - RF_N_REPEATS: how many repeats per mode to reduce noise (default 3)
    - RF_N_JOBS_MULTI: n_jobs for multi-threaded run (default os.cpu_count() or 4)
    - RF_REQUIRE_SPEEDUP: if "1", assert multi median < threshold * single median (default "0")
    - RF_THREAD_SPEEDUP_THRESHOLD: numeric threshold (default 0.85)
    """
    # Basic config from environment (with sane defaults)
    n_estimators = int(os.getenv("RF_N_ESTIMATORS", "400"))
    n_repeats = int(os.getenv("RF_N_REPEATS", "3"))
    n_jobs_multi = os.getenv("RF_N_JOBS_MULTI")
    if n_jobs_multi is None:
        # default to number of CPUs or 4 if unknown
        try:
            n_jobs_multi = os.cpu_count() or 4
        except Exception:
            n_jobs_multi = 4
    else:
        n_jobs_multi = int(n_jobs_multi)

    # Speedup control
    require_speedup = os.getenv("RF_REQUIRE_SPEEDUP", "0") == "1"
    threshold = float(os.getenv("RF_THREAD_SPEEDUP_THRESHOLD", "0.85"))

    # Validate constructor supports n_jobs
    RFClass = RandomForestRegressor
    try:
        _ = RFClass(n_estimators=1, n_jobs=1)
    except TypeError:
        pytest.skip("RandomForestRegressor constructor doesn't accept n_jobs; cannot test threading")

    # Monkeypatch to use the fake, fast-sleeping estimator
    monkeypatch.setattr(RFClass, "_estimator_class", FakeEstimator, raising=True)

    # Tiny synthetic dataset
    n_samples = 200
    n_features = 10
    X = np.zeros((n_samples, n_features), dtype=float)
    y = np.zeros(n_samples, dtype=float)

    single_times = []
    multi_times = []

    # Warm-up single-thread (1 run, not included in timings) to let any one-time overheads occur
    rf_warm = RFClass(n_estimators=4, random_state=0, n_jobs=1)
    rf_warm.fit(X, y)

    # Measure single-threaded
    for i in range(n_repeats):
        rf = RFClass(n_estimators=n_estimators, random_state=42 + i, n_jobs=1)
        t0 = perf_counter()
        rf.fit(X, y)
        t_single = perf_counter() - t0
        single_times.append(t_single)
        print(f"[single] repeat {i+1}/{n_repeats}: {t_single:.3f}s")

    # Measure multi-threaded
    for i in range(n_repeats):
        rf = RFClass(n_estimators=n_estimators, random_state=999 + i, n_jobs=n_jobs_multi)
        t0 = perf_counter()
        rf.fit(X, y)
        t_multi = perf_counter() - t0
        multi_times.append(t_multi)
        print(f"[multi n_jobs={n_jobs_multi}] repeat {i+1}/{n_repeats}: {t_multi:.3f}s")

    med_single = median(single_times)
    med_multi = median(multi_times)

    print("\n=== RF threading benchmark ===")
    print(f"n_estimators = {n_estimators}")
    print(f"repeats = {n_repeats}")
    print(f"single median = {med_single:.3f}s")
    print(f"multi (n_jobs={n_jobs_multi}) median = {med_multi:.3f}s")
    if med_single > 0:
        print(f"median speedup: {med_multi / med_single:.3f}x (smaller is faster)")
    print("================================\n")

    assert False
    # Optionally assert that multi is faster by threshold
    if require_speedup:
        assert med_multi < med_single * threshold, (
            f"Threaded fit not sufficiently faster: single_median={med_single:.3f}s, "
            f"multi_median={med_multi:.3f}s, expected multi < {threshold*100:.0f}% of single."
        )