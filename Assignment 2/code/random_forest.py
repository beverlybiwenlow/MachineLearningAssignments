from random_tree import RandomTree
import numpy as np
from scipy.stats import mode


class RandomForest:

    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        for i in range(self.num_trees):
            model = RandomTree(max_depth=self.max_depth)
            model.fit(X, y)
            self.models.append(model)

    def predict(self, X):
        M, D = X.shape
        pred = np.zeros((self.num_trees, M))
        for idx, model in enumerate(self.models):
            pred[idx, :] = model.predict(X)
        return mode(pred)[0]
