import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import tree
from sklearn import metrics
import graphviz

#  Tried some stuff myself here
# def calculateEntropy(datapoints):
#    entropy = 0
#    n = len(datapoints)
#    for label in range(10):
#        ni = np.count_nonzero(datapoints == label)
#        if ni != 0:
#            entropy += ni/n * math.log2(ni/n)
#
#    return -entropy


# def calculateClassMix(datapoints, property):
#    classMix = 0
#    n = len(datapoints)
#    for i in range(2):
#        nl = np.count_nonzero(datapoints[:, property + 1] == i)
#        entropy = calculateEntropy(datapoints[datapoints[:, 0] == i, :])
#        classMix += nl/n * entropy

#    return classMix


# def calculateInfoGain(datapoints, property):
#    return calculateEntropy(datapoints) - calculateClassMix(datapoints, property)


if __name__ == "__main__":
    data = np.loadtxt("mfeat-pix.txt", dtype='i')  # 2000 rows, 240 columns

    trainIndices = [*range(0, 100), *range(200, 300), *range(400, 500), *range(600, 700), *range(800, 900),
                    *range(1000, 1100), *range(1200, 1300), *range(1400, 1500), *range(1600, 1700), *range(1800, 1900)]
    testIndices = [index + 100 for index in trainIndices]
    trainSamples = data[trainIndices]  # 1000 rows, 240 columns
    testSamples = data[testIndices]

    b = np.ones(100, dtype=int)
    labels = np.concatenate((np.zeros(100, dtype=int), b, b*2, b*3, b*4, b*5, b*6, b*7, b*8, b*9))

    meanTrainImages = np.zeros((240, 10))   # Generate the mean numbers
    for i in range(10):
        for j in range(240):
            meanTrainImages[j, i] = np.mean(trainSamples[i*100:(i+1)*100, j])

    trainFeatures = np.zeros((1000, 10), dtype=int)
    for i in range(1000):  # Create the training feature vectors using the mean numbers and the training samples
        for j in range(10):
            trainFeatures[i, j] = np.dot(trainSamples[i], meanTrainImages[:, j])

    testFeatures = np.zeros((1000, 10), dtype=int)
    for i in range(1000):  # Create the testing feature vectors using the mean numbers and the testing samples
        for j in range(10):
            testFeatures[i, j] = np.dot(testSamples[i], meanTrainImages[:, j])

    X = trainFeatures   # The training set, which was transformed to feature vectors
    Y = labels          # The classes of the training set
    clf = tree.DecisionTreeClassifier(max_depth=15)  # Reduce overfitting
    clf = clf.fit(X, Y)     # Make the tree

    fig, ax = plt.subplots(figsize=(10, 10))    # Plot the tree
    tree.plot_tree(clf)
    plt.show()

    dot_data = tree.export_graphviz(clf, out_file=None)    # Save the tree
    graph = graphviz.Source(dot_data)
    graph.render("number")

    predictions = clf.predict(testFeatures)  # Predict the classes of the testing set
    print("Accuracy:", metrics.accuracy_score(labels, predictions))  # Compare the predicted and the true classes
