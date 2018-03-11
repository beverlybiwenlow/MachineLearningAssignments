import numpy as np
from utils import euclidean_dist_squared


class Kmedians:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]

        while True:
            y_old = y
            # Compute euclidean distance to each mean
            dist2 = np.sqrt(euclidean_dist_squared(X, means))
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            # Update means
            for kk in range(self.k):
                means[kk] = np.median(X[y == kk], axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))
            # self.error(X, means)
            # Stop if no point changed cluster
            if changes == 0:
                break

        self.means = means

    def predict(self, X):
        means = self.means
        dist2 = np.sqrt(euclidean_dist_squared(X, means))
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)

    def error(self, X, means=None):
        if means is None:
            means = self.means
        dist = np.sqrt(euclidean_dist_squared(X, means))
        minVal = np.amin(dist, axis=1)
        # print(np.sum(minVal))
        return np.sum(minVal)
