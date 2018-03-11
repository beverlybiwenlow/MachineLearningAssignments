"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        # print('k = %i' % self.k)
        # distArray = utils.euclidean_dist_squared(self.X, Xtest)

        # sidx = distArray.argsort(axis=0)
        # arrangedDistArray = distArray[sidx, np.arange(sidx.shape[1])]
        
        # y_pred = []
        # for i in range(arrangedDistArray.shape[1]):
        #     targets = []
        #     for j in range(self.k):

        #         dist = arrangedDistArray[j][i]
        #         index = np.nonzero(distArray[:,i] == dist)[0]
        #         targets.append(self.y[index])
            
        #     targetNPArray = np.array(targets)
        #     y_pred.append(utils.mode(targetNPArray))
        distances = utils.euclidean_dist_squared(self.X, Xtest)
        sorted_indexes = np.argsort(distances, axis = 0)
        sorted_indexes = sorted_indexes[:self.k,:]
        y_pred = self.y[sorted_indexes]
        y_pred = stats.mode(y_pred)[0]

        return y_pred


class CNN(KNN):

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : an N by D numpy array
        y : an N by 1 numpy array of integers in {1,2,3,...,c}
        """

        Xcondensed = X[0:1,:]
        ycondensed = y[0:1]

        for i in range(1,len(X)):
            x_i = X[i:i+1,:]
            dist2 = utils.euclidean_dist_squared(Xcondensed, x_i)
            inds = np.argsort(dist2[:,0])
            yhat = utils.mode(ycondensed[inds[:min(self.k,len(Xcondensed))]])

            if yhat != y[i]:
                Xcondensed = np.append(Xcondensed, x_i, 0)
                ycondensed = np.append(ycondensed, y[i])

        self.X = Xcondensed
        self.y = ycondensed
        print(len(Xcondensed))