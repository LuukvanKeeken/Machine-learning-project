import numpy as np
from numpy.typing import _128Bit
from scipy.sparse import data
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn import tree
from re import I
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
import math
from HCFeatures import HCFeatures
from create_data import DataSets
from subprocess import call

def gridSearch(trainSet, labels):
    hyperparameterSpace = {'n_estimators': [10, 50, 100, 250, 500], 
                           'criterion': ["gini", "entropy"],
                           'max_depth': [10, 50, 100, None],
                           'min_samples_split': [2, 4, 6, 8, 10],
                           'min_samples_leaf': [1, 2, 3, 4, 5],
                           'max_features': ["auto", "log2", 2, 5, 9],
                           'max_leaf_nodes': [None, 10, 25, 50]}
    randomForest = RandomForestClassifier(n_jobs=6)
    gridSearch = GridSearchCV(randomForest, param_grid=hyperparameterSpace, scoring='accuracy', cv=10, verbose=2, n_jobs=6)
    gridSearch.fit(trainSet, labels)
    print("Best parameters:", gridSearch.best_params_)
    print("Best mean accuracy:", gridSearch.best_score_)

if __name__ == "__main__":
    createData = DataSets()
    dataSet = createData.digits_standard()
    trainSet = createData.training_data
    trainLabels = createData.training_labels

    features = HCFeatures()
    features.trainMeanImages(dataSet)

    trainingFeatureVectors = np.zeros((len(trainSet), 18))

    for i in range(len(trainSet)):
        trainingFeatureVectors[i] = features.predict(trainSet[i])
 
    clf = RandomForestClassifier(n_estimators=250, bootstrap=True, oob_score=True, criterion="gini", max_depth=50, max_features="log2",max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, random_state=2693, n_jobs=6)
    scores = cross_val_score(clf, trainingFeatureVectors, trainLabels, cv = 10, n_jobs=6)
    print("10-fold cross validation score:", scores.mean())

    ''' Uncomment this to find the best random state for the model
    maxScore = scores.mean()
    maxStd = scores.std()
    bestRandom = 0
    # Find the best randomState
    for i in range(0, 10000):
        clf = RandomForestClassifier(n_estimators=250, bootstrap=True, oob_score=True, criterion="gini", max_depth=50, max_features="log2",max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, random_state=i, n_jobs=6)
        scores = cross_val_score(clf, trainingFeatureVectors, trainLabels, cv = 10, n_jobs=6)
        if scores.mean() > maxScore or (scores.mean() == maxScore and scores.std() < maxStd):
            maxScore = scores.mean()
            maxStd = scores.std()
            bestRandom = i
            print("New Best accuracy:", maxScore, "New Best Random state:", bestRandom)

    print("Best accuracy:", maxScore, "Random state:", bestRandom)
    '''
    # gridSearch(trainingFeatureVectors, trainLabels)

    # Calculate importance of the features for the training set
    '''
    featureImportances = np.zeros(18)
    
    
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(trainingFeatureVectors, trainLabels)
        clf.fit(X_train, y_train)
        featureImportances += clf.feature_importances_

    print(featureImportances/10)
    '''

    # Try the model on the testing set
    dataSet = createData.digits_testing()
    testSet = createData.test_data
    testLabels = createData.test_labels

    testFeatureVectors = np.zeros((len(testSet), 18))

    for i in range(len(testSet)):
         testFeatureVectors[i] = features.predict(testSet[i])

    clf2 = RandomForestClassifier(n_estimators=250, bootstrap=True, oob_score=True, criterion="gini", max_depth=50, max_features="log2",max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, n_jobs=6, random_state=2693)
    clf2.fit(trainingFeatureVectors, trainLabels)
    predictions = clf2.predict(testFeatureVectors)

    # Print the missclassified numbers
    for i in range(len(predictions)):
        if testLabels[i] != predictions[i]:
            print(f"{testLabels[i]} misclassified as a {predictions[i]}")
            digit = np.reshape(testSet[i], (16,15))
            plt.title(f"{int(testLabels[i])} misclassified as a {predictions[i]}")
            ax = sns.heatmap(digit, cmap='gray_r')
            plt.show()

    # Print the accuracy
    print("Accuracy:", metrics.accuracy_score(testLabels, predictions))

    # Plot the confusion matrix
    cm = confusion_matrix(testLabels, predictions)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    disp.ax_.set_title("Random Forest confusion matrix")
    plt.show()
    
    # Plot part of tree 125
    tree.plot_tree(clf2.estimators_[125], feature_names = ["Horizontal Symmetry x = 3", "Horizontal Symmetry x = 8", "Islands", "Laplacian", "Fourier", "Vertical Poly Row", "Mixture of Gaussians", "Mean Shade", "Similarity to 0", "Similarity to 1", "Similarity to 2", "Similarity to 3", "Similarity to 4", "Similarity to 5", "Similarity to 6", "Similarity to 7", "Similarity to 8", "Similarity to 9"], class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], filled = True, max_depth=2, fontsize=15)
    plt.show()