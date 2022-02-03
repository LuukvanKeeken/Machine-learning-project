import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics



class FeatureMaker:
    def __init__(self, image, trainSamples, testSamples):
        self.image = image
        self.trainSamples = trainSamples
        self.testSamples = testSamples
        self.trainLength = len(trainSamples)
        self.testLength = len(testSamples)
        self.meanImages = np.zeros((240, 10))

    
    def createMeanImages(self, trainingImages):
        for i in range(10):
            for j in range(240):
                self.meanImages[j, i] = np.mean(trainingImages[i * 100:(i + 1) * 100, j])
                
    def meanNumberFeature(self):
        trainFeatures = np.zeros((self.trainLength, 10), dtype=int)
        for i in range(self.trainLength):  # Create the training feature vectors using the mean numbers and the training samples
            for j in range(10):
                trainFeatures[i, j] = np.dot(self.trainSamples[i], self.meanImages[:, j])

        testFeatures = np.zeros((self.testLength , 10), dtype=int)
        for i in range(self.testLength):  # Create the testing feature vectors using the mean numbers and the testing samples
            for j in range(10):
                testFeatures[i, j] = np.dot(self.testSamples[i], self.meanImages[:, j])

        return trainFeatures, testFeatures

    def islandsFeature(self):
        trainFeatures = np.zeros(self.trainLength, dtype=int)
        testFeatures = np.zeros(self.testLength, dtype=int)
        for i in range(self.trainLength):
            self.image = np.zeros((16, 15), dtype=int)
            count = 0
            idx = 0
            for row in range(16):
                for col in range(15):
                    self.image[row, col] = self.trainSamples[i, idx]
                    idx += 1

            for row in range(16):
                for col in range(15):
                    if self.image[row, col] == 0:
                        self.DFS(row, col)
                        count += 1

            trainFeatures[i] = count

        for i in range(self.testLength):
            self.image = np.zeros((16, 15), dtype=int)
            count = 0
            idx = 0
            for row in range(16):
                for col in range(15):
                    self.image[row, col] = self.testSamples[i, idx]
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
        trainFeatures = np.zeros(self.trainLength, dtype=int)
        for i in range(self.trainLength):  # Create the training feature vectors using the mean numbers and the training samples
            trainFeatures[i] = np.average(self.trainSamples[i])

        testFeatures = np.zeros(self.testLength, dtype=int)
        for i in range(self.testLength):  # Create the testing feature vectors using the mean numbers and the testing samples
            testFeatures[i] = np.average(self.testSamples[i])

        trainFeatures = trainFeatures.reshape(-1, 1)
        testFeatures = testFeatures.reshape(-1, 1)
        return trainFeatures, testFeatures

    def createFeatureVectors(self):
        numberTrainFeatures, numberTestFeatures = self.meanNumberFeature()
        shadeTrainFeatures, shadeTestFeatures = self.meanShadeFeature()
        islandTrainFeatures, islandTestFeatures = self.islandsFeature()
        trainFeatures = np.zeros((self.trainLength, 12), dtype=int)
        testFeatures = np.zeros((self.testLength, 12), dtype=int)

        for i in range(self.trainLength):
            for j in range(12):
                if j == 0:
                    trainFeatures[i, j] = shadeTrainFeatures[i]
                elif j == 1:
                    trainFeatures[i, j] = islandTrainFeatures[i]
                else:
                    trainFeatures[i, j] = numberTrainFeatures[i, j - 2]

        for i in range(self.testLength):
            for j in range(12):
                if j == 0:
                    testFeatures[i, j] = shadeTestFeatures[i]
                elif j == 1:
                    testFeatures[i, j] = islandTestFeatures[i]
                else:
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


def gridSearch(trainSet, labels):
    hyperparameterSpace = {'n_estimators': [10, 50, 100, 250, 500], 
                           'criterion': ["gini", "entropy"],
                           'max_depth': [10, 50, 100, None],
                           'min_samples_split': [2, 4, 6, 8, 10],
                           'min_samples_leaf': [1, 2, 3, 4, 5],
                           'max_features': ["auto", "log2", 2, 5, 10],
                           'max_leaf_nodes': [None, 10, 25, 50],
                           'class_weight': ["balanced", "balanced_subsample", None]}
    randomForest = RandomForestClassifier(n_jobs=1)
    gridSearch = GridSearchCV(randomForest, param_grid=hyperparameterSpace, scoring='accuracy', cv=10, verbose=3, n_jobs=1)
    gridSearch.fit(trainSet, labels)
    print("Best parameters:", gridSearch.best_params_)
    print("Best mean accuracy:", gridSearch.best_score_)

if __name__ == "__main__":
    trainSet, validationSet, labels = processData(np.loadtxt("mfeat-pix.txt", dtype='i'))  # 2000 rows, 240 columns

    featureMaker = FeatureMaker(np.zeros((15,16), dtype=int), trainSet, [])

    featureMaker.createMeanImages(trainSet)

    trainSet, _ = featureMaker.createFeatureVectors()

    clf = RandomForestClassifier(n_estimators=100, bootstrap=True, oob_score=True)  # 100 trees in the forest

    gridSearch(trainSet, labels)

    #scores = cross_val_score(clf, trainSet, labels, cv = 10)
    #print(scores)
    #print("Mean accuracy:", scores.mean())
    #print("Standard devtiation:", scores.std())
    #print("Accuracy:", metrics.accuracy_score(y_test, predictions))  # Compare the predicted and the true classes
