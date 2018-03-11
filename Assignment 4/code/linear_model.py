import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils


class logReg:
    # Logistic Regression
    def __init__(self, verbose=2, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
        print('ran')

    def predict(self, X):
        return np.sign(X@self.w)


class logRegL2:
    # Logistic Regression
    def __init__(self, verbose=2, lammy=1, maxEvals=100):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.bias = True
        print(verbose)

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)
        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + (self.lammy / 2) * ((np.linalg.norm(w))**2)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy * w

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        return np.sign(X@self.w)


class logRegL1:
    # Logistic Regression
    def __init__(self, verbose=2, L1_lambda=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.L1_lambda = L1_lambda
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.L1_lambda,
                                        self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        return np.sign(X@self.w)


class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape

        def minimize(ind): return findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i}
                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature
                (self.w, f) = minimize(list(selected_new))
                # error = utils.classification_error(np.sign(X[:,selected_new]@self.w), y)
                if f < minLoss - self.L0_lambda * len(selected):
                    minLoss = f
                    bestFeature = i

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes, d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X + 0.0001 * np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


class logLinearClassifier:
    def __init__(self, maxEvals=500, verbose=0):
        self.maxEvals = maxEvals
        self.verbose = verbose

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes, d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            (self.W[i], f) = findMin.findMin(self.funObj, self.W[i],
                                             self.maxEvals, X, ytmp, verbose=self.verbose)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


class softmaxClassifier:
    def __init__(self, maxEvals, verbose=0):
        self.maxEvals = maxEvals
        self.verbose = verbose

    def funObj(self, w, X, y):
        n, d = X.shape
        k = np.unique(y).size
        W = np.reshape(w, (k, d))

        # calculate the f value
        f = 0
        for idx in range(n):
            x = X[idx, :]
            ftmp = -W[y[idx]]@x.T + np.log(np.sum(np.exp(W@x.T)))
            f += ftmp

        # calculate the gradient value
        g = np.zeros((k, d))
        for c in range(k):
            for j in range(d):
                for idx in range(n):
                    x = X[idx]
                    p = np.exp(W[c, :]@x.T) / np.sum(np.exp(W@x.T))
                    g[c, j] += x[j] * (p - (c == y[idx]))
        return f, g.flatten()

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.w = np.zeros((self.n_classes * d, ))

        # solve the normal equations
        # with a bit of regularization for numerical reasons
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
        self.W = np.reshape(self.w, (self.n_classes, d))

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)
