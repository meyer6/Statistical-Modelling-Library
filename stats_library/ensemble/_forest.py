import abc
import copy
import numpy as np

import os
from concurrent.futures import ThreadPoolExecutor

from ..tree._decision import (DecisionTreeClassifier,
                              DecisionTreeRegressor,
                              )

from ..tree._xgboost import XGBoostTree

__all__ = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "AdaBoostClassifier",
    "GradientBoostingRegressor",
    "XGBoostRegressor", 
]

def clone(estimator):
    return copy.deepcopy(estimator)

class BaseRandomForest(abc.ABC):
    _estimator_class = None

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=None, n_jobs=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

        self.n_jobs = n_jobs
        if n_jobs is None:
            self.n_jobs = os.cpu_count() or 1

        self.trees = []

    def fit(self, X, y):
        self.trees = []
        self.n_classes = len(np.unique(y))
        n_samples, n_features = X.shape
        
        seeds = self.rng.integers(0, 2**31 - 1, size=self.n_estimators)

        def build_tree(seed):
            tree_rng = np.random.default_rng(seed)

            indices = tree_rng.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = self._estimator_class(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed
            )
            tree.fit(X_sample, y_sample)

            return tree
        
        if self.n_jobs == 1:
            for seed in seeds:
                self.trees.append(build_tree(seed))

        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
                self.trees = list(ex.map(build_tree, seeds))

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


class AdaBoostClassifier:
    def __init__(self, base_estimator=None, n_estimators=100, learning_rate=0.5):
        self.base_estimator = base_estimator or DecisionTreeClassifier(max_depth=1)

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.estimators_ = []
        self.estimator_weights_ = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            estimator = clone(self.base_estimator)
            
            estimator.fit(X, y, sample_weight=sample_weights)
            predictions = estimator.predict(X)
            incorrect = (predictions != y)
            error = np.sum(sample_weights[incorrect]) / np.sum(sample_weights)

            if error >= 0.5:
                break

            weight = self.learning_rate * np.log((1 - error) / (error + 1e-10))
            sample_weights *= np.exp(weight * (incorrect * 2 - 1))
            sample_weights /= np.sum(sample_weights)

            self.estimators_.append(estimator)
            self.estimator_weights_.append(weight)

        return self

    def predict(self, X):
        if not getattr(self, 'estimators_', None):
            raise ValueError("Must call 'fit' before 'predict'.")

        weighted_preds = sum(w * est.predict(X) for w, est in zip(self.estimator_weights_, self.estimators_))
        return np.sign(weighted_preds)


class GradientBoostingRegressor:
    def __init__(self, base_estimator=None, n_estimators=100, learning_rate=0.1):
        self.base_estimator = base_estimator or DecisionTreeRegressor(max_depth=3)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators_ = None
        self.avg = None

    def fit(self, X, y):
        self.estimators_ = []

        self.avg = np.mean(y)
        residuals = y - self.avg

        for _ in range(self.n_estimators):
            estimator = clone(self.base_estimator)
            estimator.fit(X, residuals)

            residuals -= self.learning_rate * estimator.predict(X)

            self.estimators_.append(estimator)

        return self

    def predict(self, X):
        if self.estimators_ is None:
            raise ValueError("Must call 'fit' before 'predict'.")
        
        return self.avg + sum(self.learning_rate * estimator.predict(X) for estimator in self.estimators_)


class XGBoostRegressor:
    def __init__(self, loss='squared_error', n_estimators=100,
                 learning_rate=0.3, max_depth=3, min_child_weight=1.0,
                 gamma=0.0, reg_lambda=1.0):
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self._grad, self._hess = self._get_grad_hess(loss)
        self.base_score = None
        self.trees = []

    def fit(self, X, y):
        self.base_score = np.mean(y) 
        y_pred = np.full_like(y, self.base_score, dtype=np.float64)

        for _ in range(self.n_estimators):
            g = self._grad(y, y_pred)
            h = self._hess(y, y_pred)

            tree = XGBoostTree(max_depth=self.max_depth,
                               min_child_weight=self.min_child_weight,
                               gamma=self.gamma,
                               reg_lambda=self.reg_lambda)
            tree.fit(X, g, h)
            self.trees.append(tree)

            y_pred += self.learning_rate * tree.predict(X)

        return self

    def predict(self, X):
        if self.trees is None:
            raise ValueError("Must call 'fit' before 'predict'.")
        
        y_pred = np.full(X.shape[0], self.base_score, dtype=np.float64)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

    def _get_grad_hess(self, loss):
        if loss == 'squared_error':
            grad = lambda y, y_pred: y_pred - y
            hess = lambda y, y_pred: np.ones_like(y_pred)
        elif loss == 'poisson':
            grad = lambda y, y_pred: np.exp(y_pred) - y
            hess = lambda y, y_pred: np.exp(y_pred)
        else:
            raise ValueError("Unsupported loss")
        return grad, hess