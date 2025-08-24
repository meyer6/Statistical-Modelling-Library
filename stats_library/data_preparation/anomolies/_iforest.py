import numpy as np
from stats_library.tree._isolation import IsolationTree

class IsolationForest:
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.estimators_ = None

    def fit(self, X):
        self.estimators_ = []

        n_samples, n_features = X.shape
        if self.max_samples == 'auto':
            self.max_samples = min(256, n_samples)
            
        rng = np.random.RandomState(self.random_state) 

        for _ in range(self.n_estimators):
            idx = rng.choice(n_samples, self.max_samples, replace=False)
            seed = rng.randint(0, np.iinfo(np.int32).max)

            tree = IsolationTree(max_depth=10, random_state=seed).fit(X[idx])
            self.estimators_.append(tree)

    
    def predict(self, X):
        if self.estimators_ is None:
            raise ValueError("IsolationForest not fitted; call `.fit(X)` first.")
    
        scores = np.zeros(X.shape[0])
        
        for tree in self.estimators_:
            scores += tree.path_length(X)
        
        scores /= self.n_estimators

        c_psi = self._c(self.max_samples)
        scores = np.power(2.0, -scores / c_psi)
        
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return np.where(scores > threshold, -1, 1)
    
    def _c(self, n):
        if n <= 1:
            return 1.0

        H = np.sum(1.0 / np.arange(1, n))
        return 2.0 * H - 2.0 * (n - 1) / n