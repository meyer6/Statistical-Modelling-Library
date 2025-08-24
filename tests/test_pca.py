import numpy as np
import pytest

# Assuming the KernelPCA implementation is in kernel_pca.py
from stats_library.decomposition._pca import KernelPCA

def test_init_defaults():
    kp = KernelPCA()
    assert kp.n_components is None
    assert kp.kernel == 'rbf'
    assert kp.gamma is None
    assert kp.degree == 3
    assert kp.coef0 == 1
    # No fit state
    assert kp.X_fit_ is None
    assert kp.alphas_ is None
    assert kp.lambdas_ is None


def test_transform_before_fit_raises():
    kp = KernelPCA()
    X_new = np.zeros((2, 2))
    with pytest.raises(RuntimeError):
        kp.transform(X_new)


def test_linear_kernel_matches_pca():
    # Generate random data
    rng = np.random.RandomState(42)
    X = rng.randn(20, 5)

    # Fit KernelPCA with linear kernel
    kp = KernelPCA(n_components=3, kernel='linear')
    kp.fit(X)
    X_kp = kp.transform(X)

    # Compare against standard PCA via SVD
    # Center X
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_pca = X_centered.dot(Vt.T[:, :3])

    # KernelPCA components may differ by sign, so compare absolute values
    assert X_kp.shape == (20, 3)
    assert np.allclose(np.abs(X_kp), np.abs(X_pca), atol=1e-6)


def test_poly_kernel_shapes():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    kp = KernelPCA(n_components=2, kernel='poly', degree=2, coef0=1)
    kp.fit(X)
    X_proj = kp.transform(X)
    # Check shapes
    assert kp.lambdas_.shape == (2,)
    assert kp.alphas_.shape == (4, 2)
    assert X_proj.shape == (4, 2)


def test_rbf_gamma_default_and_custom():
    X = np.random.RandomState(0).randn(10, 4)
    # Default gamma = 1 / n_features
    kp_def = KernelPCA(kernel='rbf')
    gamma_def = kp_def._get_gamma(X)
    assert gamma_def == pytest.approx(1 / X.shape[1])

    # Custom gamma
    kp_custom = KernelPCA(kernel='rbf', gamma=0.5)
    gamma_custom = kp_custom._get_gamma(X)
    assert gamma_custom == 0.5


def test_unsupported_kernel_raises():
    X = np.zeros((3, 3))
    kp = KernelPCA(kernel='unsupported')
    with pytest.raises(ValueError):
        kp._kernel(X, X)


def test_n_components_more_than_samples():
    X = np.random.RandomState(1).randn(5, 2)
    # Request more components than samples
    kp = KernelPCA(n_components=10, kernel='linear')
    # Should not raise in fit but slice down to available
    kp.fit(X)
    # Only min(n_samples, n_components) lambdas returned
    assert kp.lambdas_.shape[0] == 5
    assert kp.alphas_.shape == (5, 5)
