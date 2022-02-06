from re import I, L
from warnings import catch_warnings
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import math
import os
import pickle

mogModelFileName = "./mogModel/mogModel"

class HCFeatures():
    def __init__(self):
        # then to load the file
        self.mogModel = None
        self.meanImages = None

        if os.path.isfile(mogModelFileName + '.pkl') and os.path.isfile(mogModelFileName+ '_labels.npy') and os.path.isfile(mogModelFileName+ '_accuracy.npy'):
            with open(mogModelFileName + '.pkl', 'rb') as file:  
                loaded_gmm = pickle.load(file)
            labels = np.load(mogModelFileName+ '_labels.npy')
            accuracy = np.load(mogModelFileName+ '_accuracy.npy')
            self.mogModel = (loaded_gmm, labels, accuracy)
            self.trainingRequired = False
        else:
            self.trainingRequired = True

    def trainMeanImages(self, dataset):
        datasetX, _ = dataset
        # datasetX *= int(255/datasetX.max())
        # datasetX = datasetX.astype(np.uint8)
        self.meanImages = np.zeros((240, 10))
        for i in range(10):
            for j in range(240):
                self.meanImages[j, i] = np.mean(datasetX[i * 100:(i + 1) * 100, j])
                
    def trainMoG(self, dataset):
        datasetX, datasetY = dataset
        datasetX *= int(255/datasetX.max())
        datasetX = datasetX.astype(np.uint8)
        
        #gmm = templateMoG
        components = 20
        gmm = GaussianMixture(n_components=components,
                                covariance_type='full', 
                                tol=1e-10, # only effect when 0
                                reg_covar=1e-10, #default: 1e-06  best 1e-10
                                max_iter=100, 
                                n_init=20, # higher is better change on good model
                                init_params='kmeans', 
                                weights_init=None, 
                                means_init=None, 
                                precisions_init=None, 
                                random_state=None, 
                                warm_start=False, 
                                verbose=0,
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
        good = 0
        for index in range(components):
            mostCount = np.argmax(labels[index])
            good += labels[index][mostCount]
            modelLabels[index] = mostCount
        
        
        accuracy = good/1000

        if self.mogModel == None:      
            save = True
            print("MoG: no model loaded, save this one")
        elif accuracy > self.mogModel[2]:
            save = True
            print("MoG: save model with greater accuracy of " + str(accuracy))
        else:
            save = False

        if save:
            with open(mogModelFileName + '.pkl', 'wb') as file:  
                pickle.dump(gmm, file)
            self.mogModel = (gmm, modelLabels, accuracy)
            #np.save(mogModelFileName + '_weights', gmm.weights_, allow_pickle=False)
            #np.save(mogModelFileName + '_means', gmm.means_, allow_pickle=False)
            #np.save(mogModelFileName + '_covariances', gmm.covariances_, allow_pickle=False)
            np.save(mogModelFileName + '_labels', np.array(modelLabels), allow_pickle=False)
            np.save(mogModelFileName + '_accuracy', accuracy, allow_pickle=False)
        return

     # all features
    def predict(self, predictX):
        defaultImage = predictX.copy()
        image = np.reshape(predictX, (16,15))
        reshapedSameColor = image.copy()
        image *= int(255/image.max())
        image.astype(np.uint8)
        image = (255-image)
        featureVector = []
        
        featureVector.append(self.featureVerticalRatio(image.copy(), yParameter = 3))
        featureVector.append(self.featureVerticalRatio(image.copy(), yParameter = 8))
        featureVector.append(self.featureIslands(reshapedSameColor.copy()))
        featureVector.append(self.featureLaplacian(image.copy()))
        featureVector.append(self.featureFourier(image.copy()))
        featureVector.append(self.featureVerticalPolyRow(image.copy()))
        featureVector.append(self.featureMoG(image.copy()))
        featureVector.append(self.featureMeanBrightness(image.copy()))

        #featureVector.extend(self.featurePrototypeMatching(image.copy()))
        featureVector.extend(self.featurePrototypeMatching(defaultImage.copy()))
        
        return featureVector
    
    def featureVerticalRatio(self, image, yParameter):
        image[image >= 20] = 255
        image[image < 20] = 0
        height, width = np.shape(image)
        pixelsTop = 0
        pixelsDown = 0
        for i in range(height):
            for j in range(width):
                if image[i,j] == 0:
                    if i < yParameter:
                        pixelsTop += 1
                    else:
                        pixelsDown += 1
        total = pixelsTop + pixelsDown
        return pixelsTop / total

    def featureIslands(self, image):
        # add extra boarder to image
        width, height = np.shape(image)
        #column = 255 * np.ones(height)
        column = 0 * np.ones(height)
        #row = 255 * np.ones((width+2,1))
        row = 0 * np.ones((width+2,1))
        image = np.vstack((column,image))
        image = np.vstack((image,column))       
        image = np.hstack((image, row))
        image = np.hstack((row,image))
        image = image.astype(np.uint8)
        
        #threshold = 100
        threshold = 3
        image[image < threshold] = 10
        image[image < 10] = 0
        image[image == 10] = 1
        self.graph = image.copy()
        
        count = 0
        for i in range(len(image)):
            for j in range(len(image[0])):
                # If a cell with value 1 is not visited yet, then new island found
                if self.graph[i][j] == 1:
                    # Visit all cells in this island and increment island count
                    self.DFS(i, j)
                    count += 1
                    #islandLocation = (i, j)
        result = count-1
        # if count == 1:
        #     result = 0
        # elif count == 2:
        #     result = 0.5
        #     islandLocation
        # else:
        #     result = 1
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
        # ratio -=0.2
        # ratio *=2
        return ratio

    def featureFourier(self, image):
        dft = cv2.dft(np.float32(image),flags = cv2.DFT_COMPLEX_OUTPUT)
        #magnitude_spectrum = 20*np.log(cv2.magnitude(dft[:,:,0],dft[:,:,1])+1e-15)
        magnitude_spectrum = np.log(1+cv2.magnitude(dft[:,:,0],dft[:,:,1]))
        #result = float(np.average(dft[:,:,0]))
        result = float(np.average(magnitude_spectrum))
        #result -=6
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
        verticalResult = np.arctan(verticalModel[0])#/1.5
        return verticalResult
    
    def featureDiagonalUp(self, image):
        height, width  = np.shape(image)
        horizontalValues = np.zeros(width)
        for i in range(width):
            xPixel = 0 + i
            yPixel = (height-1) - i
            horizontalValues[i] = image[yPixel, xPixel]
        result = np.average(horizontalValues)#/250
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
        result = np.dot(downValues, upValues)#/600000
        return result

    def featureMoG(self, image):
        gmm, modelLabels, _ = self.mogModel
        image = (255-image)
        image = image.reshape(1,-1)
        prediction = modelLabels[int(gmm.predict(image))]
        return prediction#/10

    def featureMeanBrightness(self, image):
        result = np.average(image)
        # result -= 66
        # result *=0.01
        return result

    def featurePrototypeMatching(self, image):
        image = image.flatten()
        meanFeatureVector = np.zeros((10))
        for digit in range(10):
            meanFeatureVector[digit] = np.dot(image, self.meanImages[:, digit])
        return meanFeatureVector