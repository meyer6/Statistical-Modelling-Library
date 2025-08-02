import numpy as np

__all__ = [
    "PCA",
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
