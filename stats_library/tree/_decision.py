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


class BaseDecisionTree:
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

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X)) / len(X)

        self._init_task(y)
        self._tree = self._build_tree(X, y, sample_weight)
        return self

    def predict(self, X):
        return np.array([self._predict(self._tree, x) for x in X])

    def _build_tree(self, X, y, sw, depth=0):
        num_samples = len(y)

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or num_samples < self.min_samples_split
            or self._is_pure(y)
        ):
            return _DecisionNode(value=self._leaf_value(y, sw))

        feature, threshold = self._get_best_split(X, y, sw)

        if feature is None:
            return _DecisionNode(value=self._leaf_value(y, sw))

        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        left = self._build_tree(X[left_idx], y[left_idx], sw[left_idx], depth=depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], sw[right_idx], depth=depth + 1)

        return _DecisionNode(feature_index=feature, threshold=threshold, left=left, right=right)

    def _get_best_split(self, X, y, sw):
        _, n_features = X.shape
        features = self._get_candidate_features(n_features)

        best_gain, best_feature, best_threshold = 0, None, None
        for feature in features:
            for thresh in np.unique(X[:,feature]):
                left_idx = X[:, feature] <= thresh
                right_idx = ~left_idx

                if (
                    left_idx.sum() < self.min_samples_leaf
                    or right_idx.sum() < self.min_samples_leaf
                ):
                    continue

                gain = self._split_gain(y, sw, left_idx, right_idx)

                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, thresh

        return best_feature, best_threshold

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

    def _predict(self, node, X):
        if node.is_leaf_node():
            return node.value

        if X[node.feature_index] <= node.threshold:
            return self._predict(node.left, X)

        return self._predict(node.right, X)

    # Placeholder methods — must be implemented in subclass
    def _init_task(self, y):
        raise NotImplementedError

    def _leaf_value(self, y, sw):
        raise NotImplementedError

    def _split_gain(self, y, sw, left_idx, right_idx):
        raise NotImplementedError

    def _is_pure(self, y):
        raise NotImplementedError


class DecisionTreeClassifier(BaseDecisionTree):
    def _init_task(self, y):
        self.n_classes = len(np.unique(y))

    def predict_proba(self, X):
        if self._tree is None:
            raise ValueError("Cannot predict before fitting")
        return np.array([self._predict(self._tree, x) for x in X])

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def _leaf_value(self, y, sw):
        return np.bincount(y, weights=sw, minlength=self.n_classes) / sw.sum()

    def _split_gain(self, y, sw, left_idx, right_idx):
        def gini(arr, w):
            p = np.bincount(arr, weights=w, minlength=self.n_classes) / w.sum()
            return 1 - np.sum(p ** 2)

        total_impurity = gini(y, sw)
        w_l = sw[left_idx].sum()
        w_r = sw[right_idx].sum()
        w_tot = sw.sum()
        return total_impurity - (w_l / w_tot * gini(y[left_idx], sw[left_idx]) + w_r / w_tot * gini(y[right_idx], sw[right_idx]))

    def _is_pure(self, y):
        return len(np.unique(y)) <= 1


class DecisionTreeRegressor(BaseDecisionTree):
    def _init_task(self, y):
        pass

    def _leaf_value(self, y, sw):
        return np.average(y, weights=sw)

    def _split_gain(self, y, sw, left_idx, right_idx):
        def mse(arr, w):
            avg = np.average(arr, weights=w)
            return np.average((arr - avg) ** 2, weights=w)

        total_error = mse(y, sw)
        w_l = sw[left_idx].sum()
        w_r = sw[right_idx].sum()
        w_tot = sw.sum()
        return total_error - (w_l / w_tot * mse(y[left_idx], sw[left_idx]) + w_r / w_tot * mse(y[right_idx], sw[right_idx]))

    def _is_pure(self, y):
        return np.all(y == y[0])



# class _DecisionNode:
#     def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
#         self.feature_index = feature_index  # index of the feature used for splitting
#         self.threshold = threshold          # threshold value to split on
#         self.left = left                    # left child node
#         self.right = right                  # right child node
#         self.value = value                  # stored value (class probabilities or regression output)

#     def is_leaf_node(self):
#         return self.value is not None


# class DecisionTreeClassifier:
#     def __init__(
#         self,
#         *,
#         max_depth=None,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         max_features=None,
#         random_state=None
#     ):
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.max_features = max_features
#         self.rng = np.random.RandomState(random_state)
#         self._tree = None
#         self.n_classes = None

#     def fit(self, X, y, sample_weight=None):
#         self.n_classes = len(np.unique(y))

#         if sample_weight is None:
#             sample_weight = np.ones(len(X)) / len(X)

#         self._tree = self._build_tree(X, y, sample_weight)

#         return self
    
#     def predict_proba(self, X):
#         if self._tree is None:
#             raise ValueError("Cannot predict before fitting")
        
#         return np.array([self._predict(self._tree, x) for x in X])

#     def predict(self, X):
#         proba = self.predict_proba(X)
#         return np.argmax(proba, axis=1)
    
#     def _build_tree(self, X, y, sw, depth=0):
#         num_samples = len(y)

#         if (
#             (self.max_depth is not None and depth >= self.max_depth)
#             or num_samples < self.min_samples_split
#             or len(np.unique(y)) <= 1
#         ):
#             return _DecisionNode(value=np.bincount(y, weights=sw, minlength=self.n_classes) / sw.sum())
        
#         feature, threshold = self._get_best_split(X, y, sw)

#         if feature is None:
#             return _DecisionNode(value=np.bincount(y, weights=sw, minlength=self.n_classes) / sw.sum())
        
#         left_idx = X[:, feature] <= threshold
#         right_idx = ~left_idx

#         left = self._build_tree(X[left_idx], y[left_idx], sw[left_idx], depth=depth + 1)
#         right = self._build_tree(X[right_idx], y[right_idx], sw[right_idx], depth=depth + 1)

#         return _DecisionNode(feature_index=feature, threshold=threshold, left=left, right=right)
    
#     def _get_best_split(self, X, y, sw):
#         _, n_features = X.shape
#         features = self._get_candidate_features(n_features)

#         best_gain, best_feature, best_threshold = 0, None, None
#         for feature in features:
#             for thresh in np.unique(X[:,feature]):
#                 left_idx = X[:, feature] <= thresh
#                 right_idx = ~left_idx

#                 if (
#                     left_idx.sum() < self.min_samples_leaf
#                     or right_idx.sum() < self.min_samples_leaf
#                 ):
#                     continue

#                 gain = self._split_gain(y, sw, left_idx, right_idx)

#                 if gain > best_gain:
#                     best_gain, best_feature, best_threshold = gain, feature, thresh

#         return best_feature, best_threshold

#     def _get_candidate_features(self, n_features):
#         if self.max_features is None:
#             return list(range(n_features))
#         if isinstance(self.max_features, int):
#             k = self.max_features
#         elif isinstance(self.max_features, float):
#             k = max(1, int(self.max_features * n_features))
#         elif self.max_features == "sqrt":
#             k = max(1, int(np.sqrt(n_features)))
#         elif self.max_features == "log2":
#             k = max(1, int(np.log2(n_features)))
#         else:
#             raise ValueError(f"Invalid max_features: {self.max_features}")
        
#         return list(self.rng.choice(n_features, k, replace=False))


#     def _split_gain(self, y, sw, left_idx, right_idx):
#         # Gini impurity gain with sample weights
#         def gini(arr, w):
#             p = np.bincount(arr, weights=w, minlength=self.n_classes) / w.sum()
#             return 1 - np.sum(p ** 2)
        
#         total_impurity = gini(y, sw)
#         w_l = sw[left_idx].sum()
#         w_r = sw[right_idx].sum()
#         w_tot = sw.sum()
#         return total_impurity - (w_l / w_tot * gini(y[left_idx], sw[left_idx]) + w_r / w_tot * gini(y[right_idx], sw[right_idx]))
    

#     def _predict(self, node, X):
#         if node.is_leaf_node():
#             return node.value
        
#         if X[node.feature_index] <= node.threshold:
#             return self._predict(node.left, X)
        
#         return self._predict(node.right, X)




# class _BaseDecisionTree(abc.ABC):
#     def __init__(
#         self,
#         *,
#         max_depth=None,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         max_features=None,
#         random_state=None
#     ):
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.max_features = max_features
#         self.rng = np.random.RandomState(random_state)
#         self._tree = None

#     def fit(self, X, y, sample_weight=None):
#         n_samples = X.shape[0]

#         if sample_weight is None:
#             sample_weight = np.ones(n_samples) / n_samples

#         self._tree = self._build_tree(X, y, sample_weight, depth=0)
#         return self

#     def predict(self, X):
#         return np.array([self._predict_one(x, self._tree) for x in X])

#     def _build_tree(self, X, y, sw, depth):
#         n_samples, n_features = X.shape
#         if (
#             (self.max_depth is not None and depth >= self.max_depth)
#             or n_samples < self.min_samples_split
#             or self._stop_split(y)
#         ):
#             return _DecisionNode(value=self._make_leaf(y, sw))

#         feat, thr = self._best_split(X, y, sw)
#         if feat is None:
#             return _DecisionNode(value=self._make_leaf(y, sw))

#         left_mask = X[:, feat] <= thr
#         right_mask = ~left_mask

#         left = self._build_tree(X[left_mask], y[left_mask], sw[left_mask], depth + 1)
#         right = self._build_tree(X[right_mask], y[right_mask], sw[right_mask], depth + 1)

#         return _DecisionNode(feature_index=feat, threshold=thr, left=left, right=right)

#     def _best_split(self, X, y, sw):
#         best_gain, best_feat, best_thr = -np.inf, None, None
#         n_features = X.shape[1]
#         features = self._get_candidate_features(n_features)

#         for feat in features:
#             for thr in np.unique(X[:, feat]):
#                 left_mask = X[:, feat] <= thr
#                 right_mask = ~left_mask

#                 if (
#                     left_mask.sum() < self.min_samples_leaf
#                     or right_mask.sum() < self.min_samples_leaf
#                 ):
#                     continue

#                 gain = self._split_gain(
#                     y, y[left_mask], y[right_mask],
#                     sw, sw[left_mask], sw[right_mask]
#                 )
                
#                 if gain > best_gain:
#                     best_gain, best_feat, best_thr = gain, feat, thr

#         return best_feat, best_thr

#     def _get_candidate_features(self, n_features):
#         if self.max_features is None:
#             return list(range(n_features))
#         if isinstance(self.max_features, int):
#             k = self.max_features
#         elif isinstance(self.max_features, float):
#             k = max(1, int(self.max_features * n_features))
#         elif self.max_features == "sqrt":
#             k = max(1, int(np.sqrt(n_features)))
#         elif self.max_features == "log2":
#             k = max(1, int(np.log2(n_features)))
#         else:
#             raise ValueError(f"Invalid max_features: {self.max_features}")
        
#         return list(self.rng.choice(n_features, k, replace=False))

#     def _predict_one(self, x, node):
#         if node.is_leaf_node():
#             return node.value
#         branch = node.left if x[node.feature_index] <= node.threshold else node.right
#         return self._predict_one(x, branch)

#     def _stop_split(self, y):
#         return False

#     @abc.abstractmethod
#     def _make_leaf(self, y, sw):
#         ...

#     @abc.abstractmethod
#     def _split_gain(self, y, y_left, y_right, sw, sw_left, sw_right):
#         ...

# # class DecisionTreeClassifier(_BaseDecisionTree):
# #     def __init__(self, **kwargs):
# #         super().__init__(**kwargs)
# #         self.n_classes = None

# #     def fit(self, X, y, sample_weight=None):
# #         self.n_classes = len(np.unique(y))
# #         return super().fit(X, y, sample_weight)

# #     def predict_proba(self, X):
# #         return super().predict(X)

# #     def predict(self, X):
# #         proba = self.predict_proba(X)
# #         return np.argmax(proba, axis=1)

# #     def _stop_split(self, y):
# #         # stop if one class remains
# #         return len(np.unique(y)) <= 1

# #     def _make_leaf(self, y, sw):
# #         counts = np.bincount(y, weights=sw, minlength=self.n_classes)
# #         return counts / counts.sum()

# #     def _split_gain(self, y, y_left, y_right, sw, sw_left, sw_right):
# #         # Gini impurity gain with sample weights
# #         def gini(arr, w):
# #             p = np.bincount(arr, weights=w, minlength=self.n_classes) / w.sum()
# #             return 1 - np.sum(p ** 2)
        
# #         total_impurity = gini(y, sw)
# #         w_l = sw_left.sum()
# #         w_r = sw_right.sum()
# #         w_tot = sw.sum()
# #         return total_impurity - (w_l / w_tot * gini(y_left, sw_left) + w_r / w_tot * gini(y_right, sw_right))

# class DecisionTreeRegressor(_BaseDecisionTree):
#     def _make_leaf(self, y, sw):
#         return np.average(y, weights=sw)

#     def _split_gain(self, y, y_left, y_right, sw, sw_left, sw_right):
#         # Variance reduction with sample weights
#         w_tot = sw.sum()
#         var_total = np.average((y - np.average(y, weights=sw))**2, weights=sw) if y.size > 0 else 0
#         w_l = sw_left.sum() / w_tot
#         w_r = sw_right.sum() / w_tot
#         var_l = np.average((y_left - np.average(y_left, weights=sw_left))**2, weights=sw_left) if y_left.size > 0 else 0
#         var_r = np.average((y_right - np.average(y_right, weights=sw_right))**2, weights=sw_right) if y_right.size > 0 else 0
#         return var_total - (w_l * var_l + w_r * var_r)
