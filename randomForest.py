import numpy as np
from numpy.typing import _128Bit
from scipy.sparse import data
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
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
                
    def meanNumberFeature(self): # 10 features
        # Create the training feature vectors using the mean numbers and the training samples
        trainFeatures = np.zeros((self.trainLength, 10), dtype=int)
        for i in range(self.trainLength):  
            for j in range(10):
                trainFeatures[i, j] = np.dot(self.trainSamples[i], self.meanImages[:, j])

        # Create the testing feature vectors using the mean numbers and the testing samples
        testFeatures = np.zeros((self.testLength , 10), dtype=int)
        for i in range(self.testLength): 
            for j in range(10):
                testFeatures[i, j] = np.dot(self.testSamples[i], self.meanImages[:, j])

        return trainFeatures, testFeatures

    def islandsFeature(self): # 1 feature
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

    def meanShadeFeature(self): # 1 feature
        # Create the training feature vectors using the mean numbers and the training samples
        trainFeatures = np.zeros(self.trainLength, dtype=int)
        for i in range(self.trainLength):  
            trainFeatures[i] = np.average(self.trainSamples[i])

        # Create the testing feature vectors using the mean numbers and the testing samples
        testFeatures = np.zeros(self.testLength, dtype=int)
        for i in range(self.testLength):  
            testFeatures[i] = np.average(self.testSamples[i])

        trainFeatures = trainFeatures.reshape(-1, 1)
        testFeatures = testFeatures.reshape(-1, 1)
        return trainFeatures, testFeatures

    def topCurveFeature(self, dataSet, length): # 2 features
        features = np.zeros((length, 2), dtype=int)

        for i in range(length):
            image = np.zeros((16, 15), dtype=int)
            count = 0
            idx = 0
            for row in range(16):
                for col in range(15):
                    image[row, col] = dataSet[i, idx]
                    idx += 1

            image[image >= 20] = 255
            image[image < 20] = 0
            height, width = np.shape(image)
            distance = [] 

            for i in range(width):
                for j in range(height):
                    if image[j,i] == 0 or j == height-1:
                        distance.append(j)
                        break

            highestPixel = int(np.argmin(distance))
            if highestPixel > 0:
                for i in range(highestPixel):
                    pixel = highestPixel -1 - i
                    rightPixel = highestPixel - i
                    if abs(distance[rightPixel] - distance[pixel]) > 2:
                        distance[pixel] = 100
            if highestPixel < width:
                for i in range(width - highestPixel - 1):   
                    pixel = highestPixel + i + 1
                    leftPixel = highestPixel + i      
                    if abs(distance[leftPixel]-distance[pixel]) > 2:
                        distance[pixel] = 100
        
            firstPixel = 0
            for i in range(len(distance)):
                if distance[i] < height:
                    firstPixel = i
                    break

            distance = [x for x in distance if x < height]
            while len(distance) < 3: distance.append(distance[len(distance)-1])
        
            # calculate the polynomial curve
            model = np.polyfit(range(len(distance)), distance, 2)
            lspace = np.linspace(0, len(distance)-1, len(distance))
            draw_y = np.polyval(model, lspace)   # evaluate the polynomial

            # Calculate the linear line        
            linearModel = np.polyfit(lspace, draw_y, 1, full = True)
            angle = linearModel[0][0]
            angle += 1
            angle /= 2
            residuals = linearModel[1][0]
            points = len(distance)
            result = residuals/points
            result = math.exp(result)-1

            features[i, 0] = result
            features[i, 1] = angle

        return features

    def symmetryFeature(self, dataSet, length): # 2 features
        features = np.zeros((length, 2), dtype=int)

        for i in range(length):
            image = np.zeros((16, 15), dtype=int)
            count = 0
            idx = 0
            for row in range(16):
                for col in range(15):
                    image[row, col] = dataSet[i, idx]
                    idx += 1

            image[image >= 20] = 255
            image[image < 20] = 0
            width, height = np.shape(image)
            distance = []

            pixelsLeft = 0
            pixelsRight = 0
            pixelsTop = 0
            pixelsBottom = 0

            for i in range(width):
                for j in range(height):
                    if image[i,j] == 0:
                        if i < width/2:
                            pixelsLeft += 1
                        else:
                            pixelsRight += 1
                        if j < height/2:
                            pixelsTop += 1
                        else:
                            pixelsBottom += 1
            total = pixelsLeft + pixelsRight
            verticalSymmetry = pixelsTop / total
            horizontalSymmetry = pixelsLeft / total
            
            features[i, 0] = horizontalSymmetry
            features[i, 1] = verticalSymmetry
            
        return features

    def laplacianFeature(self, dataSet, length): # 1 feature
        features = np.zeros(length, dtype=int)

        for i in range(length):
            image = np.zeros((16, 15), dtype=int)
            count = 0
            idx = 0
            for row in range(16):
                for col in range(15):
                    image[row, col] = dataSet[i, idx]
                    idx += 1

            # add extra boarder to image
            width, height = np.shape(image)
            column = 255 * np.ones(height)
            row = 255 * np.ones((width+2,1))
            image = np.vstack((column,image))
            image = np.vstack((image,column))       
            image = np.hstack((image, row))
            image = np.hstack((row,image))
            image = image.astype(np.uint8)

            # laplacian filter is sensitive to noise because it calculates the 2nd derivative. First smooth image.
            kernelSize = (5,5)
            gaussian = cv2.GaussianBlur(image,kernelSize,1)

            # frequency domain
            # Create a 2 dimensional array (3x3) and fill the array and apply the array as 2Dfilter.
            # use -1 to use the same dept as the original image
            # CustomLaplacianImg = cv2.filter2D(gaussian,-1,np.array([[1,1,1], [1,-8,1],[1,1,1]]))
            # Kernel with bad results:
            # CustomLaplacianImg = cv2.filter2D(gaussian,-1,np.array([[0,1,0], [1,-4,1],[0,1,0]]))
            # Standard laplacian filter and scale back to get rid of the visualisation image
            StandardLaplacianImg = cv2.convertScaleAbs(cv2.Laplacian(gaussian,cv2.CV_16S, 3))

            laplacianPixels = np.sum(StandardLaplacianImg)
            imageInk = np.sum(255-image)
            ratio = laplacianPixels/imageInk
            ratio -= 0.219
            ratio *= 3.116

            finaleImage = StandardLaplacianImg
            scale = 30
            width, height = np.shape(finaleImage)
            largeImage = cv2.resize(finaleImage, (width*scale, height*scale), interpolation=cv2.INTER_NEAREST)

            features[i] = ratio

        return features

    def nonWhitePixelsFeature(self, dataSet, length): # 1 feature
        features = np.zeros(length, dtype=int)

        for i in range(length):
            image = np.zeros((16, 15), dtype=int)
            count = 0
            idx = 0
            for row in range(16):
                for col in range(15):
                    image[row, col] = dataSet[i, idx]
                    idx += 1

            threshold = np.max(image)
            image[image < threshold] = 0
            image[image > 0] = 1
            totalPixels = len(image)*len(image[0])
            nonWhite = np.count_nonzero(image)
            ratio = nonWhite/totalPixels
            ratio *= 2.217
            ratio -= 0.4
            ratio = min(1,ratio)
            ratio = max(0,ratio)

            features[i] = ratio

        return features


    # The feature vectors are 18 columns long, containing the following features: 
    # [similarity to 0, similarity to 1, similarity to 2, similarity to 3, similarity to 4, similarity to 5, similarity to 6,
    #  similarity to 7, similarity to 8, similarity to 9, mean pixel value, top curve result, top curve angle, horizontal symmetry,
    #  verical symmetry, Laplacian ratio, non white pixel ratio, number of islands]
    def createFeatureVectors(self):
        numberTrainFeatures, numberTestFeatures = self.meanNumberFeature()
        shadeTrainFeatures, shadeTestFeatures = self.meanShadeFeature()
        islandTrainFeatures, islandTestFeatures = self.islandsFeature()

        trainFeatures = np.zeros((self.trainLength, 12), dtype=int)
        testFeatures = np.zeros((self.testLength, 12), dtype=int)

        for i in range(self.trainLength):
            for j in range(12):
                if j < 10:
                    trainFeatures[i, j] = numberTrainFeatures[i, j]
                elif j == 10:
                    trainFeatures[i, j] = shadeTrainFeatures[i]
                elif j == 11:
                    trainFeatures[i, j] = islandTrainFeatures[i]
                    

        for i in range(self.testLength):
            for j in range(18):
                if j < 10:
                    testFeatures[i, j] = numberTestFeatures[i, j]
                elif h == 10:
                    testFeatures[i, j] = shadeTestFeatures[i]
                elif j == 11:
                    testFeatures[i, j] = islandTestFeatures[i]

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
    testSet = createData.test_data
    trainLabels = createData.training_labels
    testLabels = createData.test_labels

    #featureMaker = FeatureMaker(np.zeros((15,16), dtype=int), trainSet, testSet)

    #featureMaker.createMeanImages(trainSet)

    #trainSet, testSet = featureMaker.createFeatureVectors()

    features = HCFeatures()
    features.fit(dataSet)

    trainingFeatureVectors = np.zeros((len(trainSet), 18))

    for i in range(len(trainSet)):
        trainingFeatureVectors[i] = features.predict(trainSet[i])

    clf = RandomForestClassifier(n_estimators=250, bootstrap=True, oob_score=True, criterion="gini", max_depth=50, max_features="log2",max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2)

    # gridSearch(trainingFeatureVectors, trainLabels)

    scores = cross_val_score(clf, trainingFeatureVectors, trainLabels, cv = 10)
    print(scores)
    print("Mean accuracy:", scores.mean())
    print("Standard devtiation:", scores.std())

    X_train, X_test, y_train, y_test = train_test_split(trainingFeatureVectors, trainLabels)
    clf.fit(X_train, y_train)
    print(clf.feature_importances_)

    testFeatureVectors = np.zeros((len(testSet), 18))

    for i in range(len(testSet)):
         testFeatureVectors[i] = features.predict(testSet[i])

    clf2 = RandomForestClassifier(n_estimators=100, bootstrap=True, oob_score=True)
    clf2.fit(trainingFeatureVectors, trainLabels)
    predictions = clf2.predict(testFeatureVectors)

    print("Accuracy:", metrics.accuracy_score(testLabels, predictions))  # Compare the predicted and the true classes
