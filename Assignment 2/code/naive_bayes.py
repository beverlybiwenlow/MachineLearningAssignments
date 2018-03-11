import numpy as np


class NaiveBayes:
    # Naive Bayes implementation.
    # Assumes the feature are binary.
    # Also assumes the labels go from 0,1,...C-1

    def __init__(self, num_classes, beta=0):
        self.num_classes = num_classes
        self.beta = beta

    def fit(self, X, y):
        N, D = X.shape

        # Compute the number of class labels
        C = self.num_classes
        beta = self.beta

        # Compute the probability of each class i.e p(y==c)
        counts = np.bincount(y)
        p_y = counts / N

        # Compute the conditional probabilities i.e.
        # p(x(i,j)=1 | y(i)==c) a
        # s p_xy
        # p(x(i,j)=0 | y(i)==c) as p_xy
        p_xy = 0.5 * np.ones((D, C))
        # TODO: replace the above line with the proper code
        for i in range(self.num_classes):
            a = (np.sum(X[y == i], axis=0) + beta) / (X[y == i].shape[0] + beta * C)
            p_xy[:, i] = a
        self.p_y = p_y
        self.p_xy = p_xy

    # This function is provided just for your understanding.
    # It should function the same as predict()
    def predict(self, X):

        N, D = X.shape
        C = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(N)
        for n in range(N):

            probs = p_y.copy()  # initialize with the p(y) terms
            for d in range(D):
                if X[n, d] != 0:
                    probs *= p_xy[d, :]
                else:
                    probs *= (1 - p_xy[d, :])

            y_pred[n] = np.argmax(probs)

        return y_pred
