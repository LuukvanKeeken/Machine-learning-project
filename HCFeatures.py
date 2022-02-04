from re import I, L
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import math

class HCFeatures():
    def __init__(self):
        self.mogModel = None
        self.meanImages = None
     
    def predict(self, predictX):
        image = np.reshape(predictX, (16,15))
        image *= int(255/image.max())
        image.astype(np.uint8)
        image = (255-image)
        featureVector = []
        
        featureVector.append(self.featureHorizontalSymmetry(image.copy(), xParameter = 3))
        featureVector.append(self.featureHorizontalSymmetry(image.copy(), xParameter = 8))
        featureVector.append(self.featureIslands(image.copy()))
        featureVector.append(self.featureLaplacian(image.copy()))
        featureVector.append(self.featureFourier(image.copy()))
        featureVector.append(self.featureVerticalPolyRow(image.copy()))
        #featureVector.append(self.featureDiagonalUp(image.copy()))
        #featureVector.append(self.featureDiagonalDot(image.copy()))
        featureVector.append(self.featureMoG(image.copy()))
        featureVector.append(self.featureMeanShade(image.copy()))
        featureVector.extend(self.featureMeanNumber(image.copy()))
        
        return featureVector
    
    # all features
    def featureHorizontalSymmetry(self, image, xParameter):
        image[image >= 20] = 255
        image[image < 20] = 0
        height, width = np.shape(image)
        pixelsLeft = 0
        pixelsRight = 0
        for i in range(height):
            for j in range(width):
                if image[i,j] == 0:
                    if i < xParameter:
                        pixelsLeft += 1
                    else:
                        pixelsRight += 1
        total = pixelsLeft + pixelsRight
        horizontalSymmetry = (pixelsLeft / total)
        return horizontalSymmetry

    def featureIslands(self, image):
        # add extra boarder to image
        width, height = np.shape(image)
        column = 255 * np.ones(height)
        row = 255 * np.ones((width+2,1))
        image = np.vstack((column,image))
        image = np.vstack((image,column))       
        image = np.hstack((image, row))
        image = np.hstack((row,image))
        image = image.astype(np.uint8)
        
        threshold = 100
        image[image < threshold] = 0
        image[image > 0] = 1
        self.graph = image.copy()
        
        count = 0
        for i in range(len(image)):
            for j in range(len(image[0])):
                # If a cell with value 1 is not visited yet, then new island found
                if self.graph[i][j] == 1:
                    # Visit all cells in this island and increment island count
                    self.DFS(i, j)
                    count += 1
                    islandLocation = (i, j)
        if count == 1:
            result = 0
        elif count == 2:
            result = 0.5
            islandLocation
        else:
            result = 1
        return result
    def DFS(self, i, j, count = -1):
        if i < 0 or i >= len(self.graph) or j < 0 or j >= len(self.graph[0]) or self.graph[i][j] != 1:
            return
        # mark it as visited
        self.graph[i][j] = count #-1
        # Recur for 8 neighbours
        self.DFS(i - 1, j - 1, count)
        self.DFS(i - 1, j, count)
        self.DFS(i - 1, j + 1, count)
        self.DFS(i, j - 1, count)
        self.DFS(i, j + 1, count)
        self.DFS(i + 1, j - 1, count)
        self.DFS(i + 1, j, count)
        self.DFS(i + 1, j + 1, count)
  
    def featureLaplacian(self, image):
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

        StandardLaplacianImg = cv2.convertScaleAbs(cv2.Laplacian(gaussian,cv2.CV_16S, 3))

        laplacianPixels = np.sum(StandardLaplacianImg)
        imageInk = np.sum(255-image)
        ratio = laplacianPixels/imageInk
        ratio -=0.2
        ratio *=2
        return ratio

    def featureFourier(self, image):
        dft = cv2.dft(np.float32(image),flags = cv2.DFT_COMPLEX_OUTPUT)
        #magnitude_spectrum = 20*np.log(cv2.magnitude(dft[:,:,0],dft[:,:,1])+1e-15)
        magnitude_spectrum = np.log(1+cv2.magnitude(dft[:,:,0],dft[:,:,1]))
        result = float(np.average(magnitude_spectrum))
        result -=6
        return result

    def featureVerticalPolyRow(self,image):
        height, width  = np.shape(image)
        verticalAverages = [0] * width
        for i in range(height-1):
            totalValue = 0
            for j in range(width):
                totalValue += image[i,j]
            verticalAverages[i] = int(totalValue/width)
        verticalModel = np.polyfit(range(len(verticalAverages)), verticalAverages, 1)   
        verticalResult = np.arctan(verticalModel[0])/1.5
        return verticalResult
    
    def featureDiagonalUp(self, image):
        height, width  = np.shape(image)
        horizontalValues = np.zeros(width)
        for i in range(width):
            xPixel = 0 + i
            yPixel = (height-1) - i
            horizontalValues[i] = image[yPixel, xPixel]
        result = np.average(horizontalValues)/250
        return result

    def featureDiagonalDot(self,image):
        height, width  = np.shape(image)

        downValues = np.zeros(width)
        for i in range(width):
            xPixel = 0 + i
            yPixel = (height-1) - i
            downValues[i] = image[yPixel, xPixel]

        upValues = np.zeros(width)
        for i in range(width):
            xPixel = 0 + i
            yPixel = i
            upValues[i] = image[yPixel, xPixel]
        result = np.dot(downValues, upValues)/600000
        return result

    def featureMoG(self, image):
        gmm, modelLabels = self.mogModel
        image = (255-image)
        image = image.reshape(1,-1)
        prediction = modelLabels[int(gmm.predict(image))]
        return prediction/10

    def featureMeanShade(self, image):
        result = np.average(image)
        result -= 66
        result *=0.01
        return result

    def featureMeanNumber(self, image):
        #trainFeatures = np.zeros((self.trainLength, 10), dtype=int)
        # for i in range(self.trainLength):  # Create the training feature vectors using the mean numbers and the training samples
        image = image.flatten()
        meanFeatureVector = np.zeros((10))
        for digit in range(10):
            meanFeatureVector[digit] = np.dot(image, self.meanImages[:, digit])
            #trainFeatures[i, j] = np.dot(self.trainSamples[i], self.meanImages[:, j])

        # testFeatures = np.zeros((self.testLength , 10), dtype=int)
        # for i in range(self.testLength):  # Create the testing feature vectors using the mean numbers and the testing samples
        #     for j in range(10):
        #         testFeatures[i, j] = np.dot(self.testSamples[i], self.meanImages[:, j])

        return meanFeatureVector

    # Fitting and training models
    def fit(self, dataset):
        self.trainMeanImages(dataset)
        self.trainMoG(dataset)

    def trainMeanImages(self, dataset):
        datasetX, _ = dataset
        self.meanImages = np.zeros((240, 10))
        for i in range(10):
            for j in range(240):
                self.meanImages[j, i] = np.mean(datasetX[i * 100:(i + 1) * 100, j])
                
    def trainMoG(self, dataset):
        datasetX, datasetY = dataset
        datasetX *= int(255/datasetX.max())
        datasetX = datasetX.astype(np.uint8)
        components = 20
        gmm = GaussianMixture(n_components=components,
                                covariance_type='full', 
                                tol=1e-10, # only effect when 0
                                reg_covar=1e-10, #default: 1e-06 
                                max_iter=100, 
                                n_init=1, # higher is better change on good model
                                init_params='kmeans', 
                                weights_init=None, 
                                means_init=None, 
                                precisions_init=None, 
                                random_state=None, 
                                warm_start=False, 
                                verbose=1,
                                verbose_interval=10)
        gmm.fit(datasetX) 
        
        # find labels by the model
        labels = np.zeros((components, 10))
        for i in range(1000):
            realLabel = datasetY[i]
            image = datasetX[i]
            image = image.reshape(1,-1)
            classPredictions = int(gmm.predict(image))
            labels[classPredictions][realLabel] += 1
        modelLabels = [0] * components
        for index in range(components):
            mostCount = np.argmax(labels[index])
            modelLabels[index] = mostCount
        self.mogModel = gmm, modelLabels
        return