import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics


class FeatureMaker:
    def __init__(self, image, trainSamples, testSamples):
        self.image = image
        self.trainSamples = trainSamples
        self.testSamples = testSamples

    def meanNumberFeature(self):
        meanTrainImages = np.zeros((240, 10))  # Generate the mean numbers
        for i in range(10):
            for j in range(240):
                meanTrainImages[j, i] = np.mean(trainSamples[i * 100:(i + 1) * 100, j])

        trainFeatures = np.zeros((1000, 10), dtype=int)
        for i in range(1000):  # Create the training feature vectors using the mean numbers and the training samples
            for j in range(10):
                trainFeatures[i, j] = np.dot(trainSamples[i], meanTrainImages[:, j])

        testFeatures = np.zeros((1000, 10), dtype=int)
        for i in range(1000):  # Create the testing feature vectors using the mean numbers and the testing samples
            for j in range(10):
                testFeatures[i, j] = np.dot(testSamples[i], meanTrainImages[:, j])

        return trainFeatures, testFeatures

    def islandsFeature(self):
        trainFeatures = np.zeros(1000, dtype=int)
        testFeatures = np.zeros(1000, dtype=int)
        for i in range(1000):
            self.image = np.zeros((16, 15), dtype=int)
            count = 0
            idx = 0
            for row in range(16):
                for col in range(15):
                    self.image[row, col] = trainSamples[i, idx]
                    idx += 1

            for row in range(16):
                for col in range(15):
                    if self.image[row, col] == 0:
                        self.DFS(row, col)
                        count += 1

            trainFeatures[i] = count

        for i in range(1000):
            self.image = np.zeros((16, 15), dtype=int)
            count = 0
            idx = 0
            for row in range(16):
                for col in range(15):
                    self.image[row, col] = testSamples[i, idx]
                    idx += 1

            for row in range(16):
                for col in range(15):
                    if self.image[row, col] == 0:
                        self.DFS(row, col)
                        count += 1

            testFeatures[i] = count

        return trainFeatures, testFeatures

    def DFS(self, i, j):
        if i < 0 or i >= 16 or j < 0 or j >= 15 or self.image[i, j] != 0:
            return

        self.image[i][j] = -1

        self.DFS(i - 1, j - 1)
        self.DFS(i - 1, j)
        self.DFS(i - 1, j + 1)
        self.DFS(i, j - 1)
        self.DFS(i, j + 1)
        self.DFS(i + 1, j - 1)
        self.DFS(i + 1, j)
        self.DFS(i + 1, j + 1)

    def meanShadeFeature(self):
        trainFeatures = np.zeros(1000, dtype=int)
        for i in range(1000):  # Create the training feature vectors using the mean numbers and the training samples
            trainFeatures[i] = np.average(trainSamples[i])

        testFeatures = np.zeros(1000, dtype=int)
        for i in range(1000):  # Create the testing feature vectors using the mean numbers and the testing samples
            testFeatures[i] = np.average(testSamples[i])

        trainFeatures = trainFeatures.reshape(-1, 1)
        testFeatures = testFeatures.reshape(-1, 1)
        return trainFeatures, testFeatures

    def createFeatureVectors(self):
        numberTrainFeatures, numberTestFeatures = self.meanNumberFeature()
        shadeTrainFeatures, shadeTestFeatures = self.meanShadeFeature()
        islandTrainFeatures, islandTestFeatures = self.islandsFeature()
        trainFeatures = np.zeros((1000, 12), dtype=int)
        testFeatures = np.zeros((1000, 12), dtype=int)

        for i in range(1000):
            for j in range(12):
                if j == 0:
                    trainFeatures[i, j] = shadeTestFeatures[i]
                    testFeatures[i, j] = shadeTrainFeatures[i]
                elif j == 1:
                    trainFeatures[i, j] = islandTestFeatures[i]
                    testFeatures[i, j] = islandTrainFeatures[i]
                else:
                    trainFeatures[i, j] = numberTrainFeatures[i, j - 2]
                    testFeatures[i, j] = numberTestFeatures[i, j - 2]

        return trainFeatures, testFeatures


def processData(data):
    trainIndices = [*range(0, 100), *range(200, 300), *range(400, 500), *range(600, 700), *range(800, 900),
                    *range(1000, 1100), *range(1200, 1300), *range(1400, 1500), *range(1600, 1700), *range(1800, 1900)]
    testIndices = [index + 100 for index in trainIndices]
    trainSamples = data[trainIndices]  # 1000 rows, 240 columns
    testSamples = data[testIndices]

    b = np.ones(100, dtype=int)
    labels = np.concatenate((np.zeros(100, dtype=int), b, b * 2, b * 3, b * 4, b * 5, b * 6, b * 7, b * 8, b * 9))

    return trainSamples, testSamples, labels


if __name__ == "__main__":
    trainSamples, testSamples, labels = processData(np.loadtxt("mfeat-pix.txt", dtype='i'))  # 2000 rows, 240 columns

    featureMaker = FeatureMaker(np.zeros((15,16), dtype=int), trainSamples, testSamples)

    trainFeatures, testFeatures = featureMaker.createFeatureVectors()

    X = trainFeatures  # The training set, which was transformed to feature vectors
    Y = labels  # The classes of the training set
    clf = RandomForestClassifier(n_estimators=100, bootstrap=True, oob_score=True)  # 500 trees in the forest
    clf = clf.fit(X, Y)  # Make the tree

    print("Out-of-bag score:", clf.oob_score_)
    predictions = clf.predict(testFeatures)  # Predict the classes of the testing set
    print("Accuracy:", metrics.accuracy_score(labels, predictions))  # Compare the predicted and the true classes
