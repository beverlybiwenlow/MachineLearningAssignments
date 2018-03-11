import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares


class LeastSquares:
    def fit(self, X, y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.


class WeightedLeastSquares(LeastSquares):  # inherits the predict() function from LeastSquares
    def fit(self, X, y, z):
        self.w = solve(X.T@z@X, X.T@z@y)


class LinearModelGradient(LeastSquares):

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w, X, y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w, X, y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' %
                  (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self, w, X, y):
        ''' MODIFY THIS CODE '''
        # Calculate the function value
        f = np.sum(np.log(np.exp(X@w - y) + np.exp(y - X@w)))
        # Calculate the gradient value
        temp = X * (np.exp(X@w - y) - np.exp(y - X@w)) / (np.exp(X@w - y) + np.exp(y - X@w))
        g = np.sum(temp, axis=0)
        return (f, g)


# Least Squares with a bias added
class LeastSquaresBias:

    def add_bias(self, X):
        ''' function which returns a new array
        with a bias column added to X '''
        bias = np.ones((X.shape[0], 1))
        return np.append(bias, X, axis=1)

    def fit(self, X, y):
        X_bias = self.add_bias(X)
        self.w = solve(X_bias.T@X_bias, X_bias.T@y)

    def predict(self, X):
        X_bias = self.add_bias(X)
        return X_bias@self.w

# Least Squares with polynomial basis


class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self, X, y):
        X_poly = self.__polyBasis(X)
        self.w = solve(X_poly.T@X_poly, X_poly.T@y)

    def predict(self, X):
        X_poly = self.__polyBasis(X)
        return X_poly@self.w

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        ''' function which returns a new numpy array
        with bias and X values raised up till power p'''
        N, d = X.shape
        X_poly = np.zeros((N, 1 + d * self.p))
        X_poly[:, 0] = np.ones((N, ))
        X_poly[:, 1: d + 1] = X
        for i in range(1, self.p):
            power = i + 1
            X_poly[:, i * d + 1:(power * d) + 1] = X**power
        return X_poly

# Least Squares with RBF Kernel


class LeastSquaresRBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self, X, y):
        self.X = X
        n, d = X.shape

        Z = self.__rbfBasis(X, X, self.sigma)

        # Solve least squares problem
        a = Z.T@Z + 1e-12 * np.identity(n)  # tiny bit of regularization
        b = Z.T@y
        self.w = solve(a, b)

    def predict(self, Xtest):
        Z = self.__rbfBasis(Xtest, self.X, self.sigma)
        yhat = Z@self.w
        return yhat

    def __rbfBasis(self, X1, X2, sigma):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        d = X1.shape[1]
        den = 1 / np.sqrt(2 * np.pi * (sigma ** 2))

        D = (X1**2)@np.ones((d, n2)) + \
            np.ones((n1, d))@(X2.T ** 2) - \
            2 * (X1@X2.T)

        Z = den * np.exp(-1 * D / (2 * (sigma**2)))
        return Z
