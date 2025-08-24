import numpy as np

__all__ = [
    "PCA",
    "KernelPCA",
]

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute the covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by explained variance
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components_ = eigenvectors[:, sorted_indices]
        self.explained_variance_ = eigenvalues[sorted_indices]

        if self.n_components is not None:
            self.components_ = self.components_[:, :self.n_components]
            self.explained_variance_ = self.explained_variance_[:self.n_components]

        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)

        return self

    def transform(self, X):
        if self.components_ is None:
            raise RuntimeError("PCA is not fitted yet.")
        
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class KernelPCA:
    def __init__(self, n_components=None, kernel="rbf", gamma=None, degree=3, coef0=1):
        self.n_components = n_components

        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

        self.X_fit_ = None
        self.alphas_ = None
        self.lambdas_ = None

        self.K_fit_ = None
        self.K_row_mean_ = None
        self.K_all_mean_ = None

    def fit(self, X):
        self.X_fit_ = np.array(X, copy=True)
        n_samples = X.shape[0]

        K = self._kernel(X, X)
        self.K_fit_ = K
        self.K_row_mean_ = np.mean(K, axis=0)
        self.K_all_mean_ = np.mean(K)

        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centre = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)


        eigenvalues, eigenvectors = np.linalg.eigh(K_centre)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if self.n_components is not None:
            eigenvalues = eigenvalues[: self.n_components]
            eigenvectors = eigenvectors[:, : self.n_components]

        self.alphas_ = eigenvectors / np.sqrt(eigenvalues)[None, :]
        self.lambdas_ = eigenvalues

        return self

    def transform(self, X):
        if self.alphas_ is None:
            raise RuntimeError("KernelPCA is not fitted yet.")

        K_test = self._kernel(X, self.X_fit_)

        # Center test kernel
        K_test_center = (
            K_test
            - self.K_row_mean_[None, :]
            - np.mean(K_test, axis=1)[:, None]
            + self.K_all_mean_
        )

        return K_test_center.dot(self.alphas_)

    def _kernel(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        
        elif self.kernel == 'poly':
            return (np.dot(X1, X2.T) + self.coef0) ** self.degree

        elif self.kernel == 'rbf':
            self.gamma = self._get_gamma(X1)

            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-sq_dists / (2 * (self.gamma ** 2)))
        
        else:
            raise ValueError("Unsupported kernel type.")
        
    def _get_gamma(self, X):
        if self.gamma is None:
            return 1.0 / X.shape[1]
        return self.gamma