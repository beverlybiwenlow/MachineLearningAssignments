# standard Python imports
import os
import argparse
import time
import pickle

# 3rd party libraries
import numpy as np                              # this comes with Anaconda
import matplotlib.pyplot as plt                 # this comes with Anaconda
import pandas as pd                             # this comes with Anaconda
from sklearn.tree import DecisionTreeClassifier # see http://scikit-learn.org/stable/install.html
from sklearn.neighbors import KNeighborsClassifier # same as above

# CPSC 340 code
import utils
from decision_stump import DecisionStump, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from knn import KNN, CNN
from simple_decision import predict

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=["1.1", "2", "2.2", "2.3", "2.4", "3", "3.1", "3.2", "4.1", "4.2", "5"])

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":
        input_file = "../data/fluTrends.csv"
        df = pd.read_csv(input_file, header = 0)

        print('Minimum: %.3f' % df.values.min())
        print('Maximum: %.3f' % df.values.max())
        print('Mean: %.3f' % df.values.mean())
        print('Median: %.3f' % np.median(df.values))
        import utils
        print('Mode: %.3f' % utils.mode(df.values))

        print('5th percentile: %.3f' % np.percentile(df.values, 5))
        print('25th percentile: %.3f' % np.percentile(df.values, 25))
        print('50th percentile: %.3f' % np.percentile(df.values, 50))
        print('75th percentile: %.3f' % np.percentile(df.values, 75))
        print('95th percentile: %.3f' % np.percentile(df.values, 95))

        means = df.mean()
        print('Highest mean is in: %s' % means.idxmax())
        print('Lowest mean is in: %s' % means.idxmin())
        variances = df.var()
        print('Highest variance is in: %s' % variances.idxmax())
        print('Lowest variance is in: %s' % variances.idxmin())
        pass
    
    elif question == "2":

        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3. Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y) 
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "2.2":
        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 3. Evaluate decision stump
        model = DecisionStump()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "2.3":
        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree
        model = DecisionTree(max_depth=2)
        model.fit(X, y)

        y_pred = model.spredict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)
    
    elif question == "2.4":
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try

        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("Our decision tree took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors, label="mine")
        
        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))

        
        plt.plot(depths, my_tree_errors, label="sklearn")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q2_4_tree_errors.pdf")
        plt.savefig(fname)
        
        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)


    elif question == "3":
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]        
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "3.1":
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]   

        depths = np.arange(1,15) # depths to try

        training_errors = np.zeros(depths.size)
        testing_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)
            print("Training error: %.3f" % tr_error)
            training_errors[i] = tr_error

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("Testing error: %.3f" % te_error)
            testing_errors[i] = te_error

        
        plt.plot(depths, training_errors, label="Training")
        plt.plot(depths, testing_errors, label="Testing")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.title("Classification error VS Depth of tree")
        plt.legend()
        fname = os.path.join("..", "figs", "training_and_testing_errors.pdf")
        plt.savefig(fname)
        pass

    elif question == "3.2":
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        size = int(len(X) / 2)
        X_train = X[:size]
        X_val = X[size:]
        y_train = y[:size]
        y_val = y[size:]

        depths = np.arange(1,15) # depths to try

        training_errors = np.zeros(depths.size)
        val_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)

            y_pred = model.predict(X_train)
            tr_error = np.mean(y_pred != y_train)
            training_errors[i] = tr_error

            y_pred = model.predict(X_val)
            te_error = np.mean(y_pred != y_val)
            val_errors[i] = te_error

        
        plt.plot(depths, training_errors, label="Training")
        plt.plot(depths, val_errors, label="Validation")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.title("Classification error VS Depth of tree")
        plt.legend()
        fname = os.path.join("..", "figs", "training_and_validation_error.pdf")
        plt.savefig(fname)
        pass

    if question == '4.1':
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"] 
        model = KNN(k = 10)
        model.fit(X, y)
        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)  


        # model2 = KNeighborsClassifier(n_neighbors = 1)
        # model2.fit(X,y)
        
        # utils.plotClassifier(model, X, y) 
        # utils.plotClassifier(model2, X, y)      
     
        pass

    if question == '4.2':
        dataset = load_dataset("citiesBig2.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]  

        # t = time.time()
        model = CNN(1)
        model.fit(X, y)
        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        # print("CNN took %f seconds" % (time.time()-t))     

        
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)  
        utils.plotClassifier(model, X, y) 


        pass
