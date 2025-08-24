import numpy as np

class IsolationTree:
    def __init__(self, max_depth=10, random_state=None):
        self.max_depth = max_depth
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)
        self.root = None

    def fit(self, X):
        n_samples, _ = X.shape
        indices = self.rng.choice(n_samples, n_samples, replace=False)
        self.root = self._build_tree(X[indices], depth=0)

        return self

    def _build_tree(self, X, depth):
        n_samples = X.shape[0]
        if n_samples <= 1 or depth >= self.max_depth:
            return None

        feature_index = self.rng.randint(X.shape[1])
        threshold = self.rng.uniform(np.min(X[:, feature_index]), np.max(X[:, feature_index]))

        left_indices = X[:, feature_index] < threshold
        right_indices = ~left_indices

        left_child = self._build_tree(X[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], depth + 1)

        return (feature_index, threshold, left_child, right_child)

    def path_length(self, X):
        return np.array([self._path_length(x, self.root) for x in X])

    def _path_length(self, x, node):
        if node is None:
            return 0

        feature_index, threshold, left_child, right_child = node
        if x[feature_index] < threshold:
            return 1 + self._path_length(x, left_child)
        else:
            return 1 + self._path_length(x, right_child)