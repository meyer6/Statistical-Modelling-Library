import abc
import numpy as np

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
]

class _DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index  # index of the feature used for splitting
        self.threshold = threshold          # threshold value to split on
        self.left = left                    # left child node
        self.right = right                  # right child node
        self.value = value                  # stored value (class probabilities or regression output)

    def is_leaf_node(self):
        return self.value is not None

class _BaseDecisionTree(abc.ABC):
    def __init__(
        self,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.RandomState(random_state)
        self._tree = None

    def fit(self, X, y):
        self._tree = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        return np.array([self._predict_one(x, self._tree) for x in X])

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or self._stop_split(y)
        ):
            return _DecisionNode(value=self._make_leaf(y))

        feat, thr = self._best_split(X, y)
        if feat is None:
            return _DecisionNode(value=self._make_leaf(y))

        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return _DecisionNode(feature_index=feat, threshold=thr, left=left, right=right)

    def _best_split(self, X, y):
        best_gain, best_feat, best_thr = -np.inf, None, None
        n_features = X.shape[1]
        features = self._get_candidate_features(n_features)

        for feat in features:
            for thr in np.unique(X[:, feat]):
                left_mask = X[:, feat] <= thr
                right_mask = ~left_mask
                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue
                gain = self._split_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, feat, thr
        return best_feat, best_thr

    def _get_candidate_features(self, n_features):
        if self.max_features is None:
            return list(range(n_features))
        if isinstance(self.max_features, int):
            k = self.max_features
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")
        
        return list(self.rng.choice(n_features, k, replace=False))

    def _predict_one(self, x, node):
        if node.is_leaf_node():
            return node.value
        branch = node.left if x[node.feature_index] <= node.threshold else node.right
        return self._predict_one(x, branch)

    def _stop_split(self, y):
        return False

    @abc.abstractmethod
    def _make_leaf(self, y):
        ...

    @abc.abstractmethod
    def _split_gain(self, y, y_left, y_right):
        ...

class DecisionTreeClassifier(_BaseDecisionTree):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        return super().fit(X, y)

    def predict_proba(self, X):
        return super().predict(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def _stop_split(self, y):
        # stop if one class remains
        return len(np.unique(y)) <= 1

    def _make_leaf(self, y):
        counts = np.bincount(y, minlength=self.n_classes)
        return counts / counts.sum()

    def _split_gain(self, y, y_left, y_right):
        # Gini impurity gain
        def gini(arr):
            p = np.bincount(arr, minlength=self.n_classes) / arr.size
            return 1 - np.sum(p ** 2)
        
        total_impurity = gini(y)
        w_l = y_left.size / y.size
        w_r = y_right.size / y.size
        return total_impurity - (w_l * gini(y_left) + w_r * gini(y_right))

class DecisionTreeRegressor(_BaseDecisionTree):
    def _make_leaf(self, y):
        return np.mean(y)

    def _split_gain(self, y, y_left, y_right):
        total_var = np.var(y) if y.size > 0 else 0
        w_l = y_left.size / y.size
        w_r = y_right.size / y.size
        return total_var - (w_l * np.var(y_left) + w_r * np.var(y_right))
