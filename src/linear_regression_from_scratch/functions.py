import numpy as np


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X_mat = np.concatenate((X.reshape(-1, 1), np.ones((*X.shape, 1))), axis=1)
            A = self._compute_matrix(X_mat, y)
            self.coef_ = A[:-1]
            self.intercept_ = A[-1]
        else:
            X_mat = X.reshape(-1, 1)
            A = self._compute_matrix(X_mat, y)
            self.coef_ = A
            self.intercept_ = "no intercept"

        self.A = A

    def predict(self, X):
        if self.fit_intercept:
            X_mat = np.concatenate((X.reshape(-1, 1), np.ones((*X.shape, 1))), axis=1)
            y = self.A.T @ X_mat.T
        else:
            X_mat = X.reshape(-1, 1)
            y = self.A.T @ X_mat.T
        return y
    
    @staticmethod
    def _compute_matrix(X_mat, y):
        Y = y.reshape(-1, 1)
        a = np.linalg.inv(X_mat.T @ X_mat)
        A = a @ X_mat.T @ Y
        return A