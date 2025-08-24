import numpy as np
from scipy.optimize import minimize

class AR:
    def __init__(self, p=1):
        self.order = p
        self.coefficients = None
        self.intercept = None
        self.last_y = None

    def fit(self, y):
        n = len(y)
        if n <= self.order:
            raise ValueError("Not enough data points to fit the model.")

        X = np.array([y[i:i + self.order] for i in range(n - self.order)])
        y_target = y[self.order:]

        A = np.column_stack([X, np.ones(len(X))])

        coeffs, _, _, _ = np.linalg.lstsq(A, y_target, rcond=None)
        self.coefficients = coeffs[:-1]
        self.intercept = coeffs[-1]

        self.last_y = y[-self.order:]

    def predict(self, steps=1):
        if self.coefficients is None:
            raise ValueError("Not enough data points to make a prediction.")

        preds = np.copy(self.last_y)
        for _ in range(steps):
            val = np.dot(preds[-self.order:], self.coefficients) + self.intercept
            preds = np.append(preds, val)

        return preds[-steps:]
    

class MA:
    def __init__(self, q):
        self.q = q
        self.mu_ = None
        self.theta_ = None
        self.residuals_ = None
    
    def fit(self, y):
        n = len(y)
        
        # Estimate mean
        self.mu_ = np.mean(y)
        
        # Approximate residuals as y - mean (first pass)
        eps_approx = y - self.mu_
        
        # Build lagged error matrix
        X = []
        for i in range(self.q):
            X.append(np.roll(eps_approx, i+1))
        X = np.array(X).T[self.q:]  # remove invalid first q rows
        
        Y = eps_approx[self.q:]
        
        # Solve OLS: Y = X * theta
        self.theta_ = np.linalg.lstsq(X, Y, rcond=None)[0]
        
        # Compute residuals for fitted model
        fitted = np.zeros_like(y)
        eps_est = np.zeros_like(y)
        for t in range(self.q, n):
            fitted[t] = self.mu_ + np.dot(self.theta_, eps_est[t-self.q:t][::-1])
            eps_est[t] = y[t] - fitted[t]
        
        self.residuals_ = eps_est
        return self
    
    def predict(self, steps=1):
        if self.mu_ is None or self.theta_ is None:
            raise ValueError("Model must be fitted before predicting.")
        
        preds = []
        eps_hist = self.residuals_.copy()
        
        for _ in range(steps):
            eps_input = np.zeros(self.q)
            pred = self.mu_ + np.dot(self.theta_, eps_input)
            preds.append(pred)
            # Append assumed residual for completeness
            eps_hist = np.append(eps_hist, 0)
        
        return np.array(preds)


class ARMA:
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.ar = AR(p)
        self.ma = MA(q)
        self.intercept = None
        self.phi = None
        self.theta = None
        self.sigma2 = None

    def fit(self, y):
        n = len(y)
        if n <= max(self.p, self.q):
            raise ValueError("Not enough data points to fit the model.")
        
        self.intercept = y.mean()
        self.ar.fit(y - self.intercept)
        self.phi = self.ar.coefficients

        eps = y - (self.intercept + self.ar.predict(y, steps=0) if self.p > 0 else 0)
        self.ma.fit(eps)
        self.theta = self.ma.coefficients
        self.sigma2 = self.ma.sigma2
        return self

    def predict(self, y, steps=1):
        if self.intercept is None:
            raise ValueError("Model must be fitted before prediction.")
        
        ar_f = self.ar.predict(y, steps)
        eps = self.ma._one_step_errors(y - self.intercept, self.theta)
        ma_f = self.ma.predict(eps, steps)
        return self.intercept + ar_f + ma_f
