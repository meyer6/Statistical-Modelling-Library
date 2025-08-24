import numpy as np

__all__ = [
    "XGBoostTree",
]

# class Node:
#     def __init__(self, depth=0, is_leaf=False, value=None, feature_index=None, threshold=None, left=None, right=None):
#         self.depth = depth
#         self.is_leaf = is_leaf
#         self.value = value
#         self.feature_index = feature_index
#         self.threshold = threshold
#         self.left = left
#         self.right = right

#     def predict(self, x):
#         if self.is_leaf:
#             return self.value
#         if x[self.feature_index] <= self.threshold:
#             return self.left.predict(x)
#         else:
#             return self.right.predict(x)


class XGBoostTree:
    def __init__(self, max_depth=3, min_child_weight=1.0,
                 gamma=0.0, reg_lambda=1.0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.root = None

    def fit(self, X, g, h):
        self.root = self._build_tree(X, g, h, depth=0)
        return self

    def _build_tree(self, X, g, h, depth):
        G = np.sum(g)
        H = np.sum(h)

        # Stop if depth limit or not enough hessian
        if depth >= self.max_depth or H < self.min_child_weight:
            leaf_value = -G / (H + self.reg_lambda)
            return Node(depth=depth, is_leaf=True, value=leaf_value)

        best_gain = -np.inf
        best_split = None

        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            not_nan_mask = ~np.isnan(X[:, feature_index])
            sorted_idx = np.argsort(X[not_nan_mask, feature_index])
            X_sorted = X[not_nan_mask][sorted_idx]
            g_sorted = g[not_nan_mask][sorted_idx]
            h_sorted = h[not_nan_mask][sorted_idx]

            G_total = np.sum(g_sorted)
            H_total = np.sum(h_sorted)

            G_left = 0.0
            H_left = 0.0

            for i in range(1, X_sorted.shape[0]):
                G_left += g_sorted[i - 1]
                H_left += h_sorted[i - 1]

                G_right = G_total - G_left
                H_right = H_total - H_left

                if H_left < self.min_child_weight or H_right < self.min_child_weight:
                    continue

                # Avoid duplicate thresholds
                if X_sorted[i, feature_index] == X_sorted[i - 1, feature_index]:
                    continue

                gain = self._calculate_gain(G_left, H_left, G_right, H_right)

                if gain > best_gain:
                    threshold = (X_sorted[i, feature_index] +
                                 X_sorted[i - 1, feature_index]) / 2
                    left_indices = np.where((X[:, feature_index] <= threshold) &
                                            ~np.isnan(X[:, feature_index]))[0]
                    right_indices = np.where((X[:, feature_index] > threshold) &
                                             ~np.isnan(X[:, feature_index]))[0]
                    nan_indices = np.where(np.isnan(X[:, feature_index]))[0]

                    best_gain = gain
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_indices': np.concatenate([left_indices, nan_indices]),
                        'right_indices': right_indices
                    }

        if best_gain < self.gamma or best_split is None:
            leaf_value = -G / (H + self.reg_lambda)
            return Node(depth=depth, is_leaf=True, value=leaf_value)

        left = self._build_tree(X[best_split['left_indices']],
                                g[best_split['left_indices']],
                                h[best_split['left_indices']],
                                depth + 1)
        right = self._build_tree(X[best_split['right_indices']],
                                 g[best_split['right_indices']],
                                 h[best_split['right_indices']],
                                 depth + 1)

        return Node(depth=depth, is_leaf=False,
                    feature_index=best_split['feature_index'],
                    threshold=best_split['threshold'],
                    left=left, right=right)

    def _calculate_gain(self, G_left, H_left, G_right, H_right):
        def term(G, H):
            return (G ** 2) / (H + self.reg_lambda)
        gain = 0.5 * (term(G_left, H_left) + term(G_right, H_right)
                      - term(G_left + G_right, H_left + H_right)) - self.gamma
        return gain

    def predict(self, X):
        if self.root is None:
            raise ValueError("Tree has not been fitted yet.")
        
        return np.array([self.root.predict(row) for row in X])