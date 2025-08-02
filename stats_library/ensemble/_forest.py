import abc

import numpy as np

from ..tree._decision import (DecisionTreeClassifier,
                              DecisionTreeRegressor)

__all__ = [
    "RandomForestClassifier",
    "RandomForestRegressor",
]

class BaseRandomForest(abc.ABC):
    _estimator_class = None

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        self.n_classes = len(np.unique(y))
        n_samples, n_features = X.shape
        
        for _ in range(self.n_estimators):
            indices = self.rng.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = self._estimator_class(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    @abc.abstractmethod
    def predict(self, X):
        ...

class RandomForestClassifier(BaseRandomForest):
    _estimator_class = DecisionTreeClassifier

    def predict_proba(self, X):
        proba = np.zeros((X.shape[0], self.n_classes))
        for tree in self.trees:
            proba += tree.predict_proba(X)
        return proba / self.n_estimators

    def predict(self, X):
        if not getattr(self, 'trees', None):
            raise ValueError("Must call 'fit' before 'predict'.")

        return np.argmax(self.predict_proba(X), axis=1)
    
class RandomForestRegressor(BaseRandomForest):
    _estimator_class = DecisionTreeRegressor

    def predict(self, X):
        if not getattr(self, 'trees', None):
            raise ValueError("Must call 'fit' before 'predict'.")

        preds = np.stack([tree.predict(X) for tree in self.trees], axis=0)
        return preds.mean(axis=0)
