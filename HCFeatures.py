import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
import os
import pickle

mogModelFileName = "./mogModel/mogModel"

class HCFeatures():
    def __init__(self):
        self.mogModel = None
        self.meanImages = None

        # check if a mixture of gaussian model exist and open it.
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
                                tol=1e-10,
                                reg_covar=1e-10,
                                max_iter=100, 
                                n_init=20,
                                init_params='kmeans', 
                                weights_init=None, 
                                means_init=None, 
                                precisions_init=None, 
                                random_state=None, 
                                warm_start=False, 
                                verbose=0,
                                verbose_interval=10)
        gmm.fit(datasetX) 
        
        # Match the labels of the MoG with the real labels. 
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
        
        # calculate the accuracy of the trained model on the training data
        accuracy = good/1000

        # If no model exist or the accuracy of the new model is better, save the new model. 
        if self.mogModel == None:      
            save = True
            print("MoG: no model loaded, save this one")
        elif accuracy > self.mogModel[2]:
            save = True
            print("MoG: save model with greater accuracy of " + str(accuracy))
        else:
            save = False

        # if save, save the model as dump, and also its labels and the found accuracy
        if save:
            with open(mogModelFileName + '.pkl', 'wb') as file:  
                pickle.dump(gmm, file)
            self.mogModel = (gmm, modelLabels, accuracy)
            np.save(mogModelFileName + '_labels', np.array(modelLabels), allow_pickle=False)
            np.save(mogModelFileName + '_accuracy', accuracy, allow_pickle=False)
        return

    # all features
    def predict(self, predictX):
        reshapedImage = np.reshape(predictX, (16,15))
        normalizedImage = reshapedImage.copy()
        normalizedImage *= int(255/normalizedImage.max())
        normalizedImage.astype(np.uint8)
        normalizedImage = (255-normalizedImage)
        
        # create feature vector for the image
        featureVector = []
        featureVector.append(self.featureVerticalRatio(normalizedImage.copy(), yParameter = 3))
        featureVector.append(self.featureVerticalRatio(normalizedImage.copy(), yParameter = 8))
        featureVector.append(self.featureIslands(reshapedImage.copy()))
        featureVector.append(self.featureLaplacian(normalizedImage.copy()))
        featureVector.append(self.featureFourier(normalizedImage.copy()))
        featureVector.append(self.featureRegressionRowAverages(normalizedImage.copy()))
        featureVector.append(self.featureMoG(normalizedImage.copy()))
        featureVector.append(self.featureMeanBrightness(normalizedImage.copy()))
        # the prototype matching feature requires the direct predicting data, and will add 10 features
        featureVector.extend(self.featurePrototypeMatching(predictX.copy()))
        
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
        column = np.zeros(height)
        row = np.zeros((width+2,1))
        image = np.vstack((column,image))
        image = np.vstack((image,column))       
        image = np.hstack((image, row))
        image = np.hstack((row,image))
        image = image.astype(np.uint8)

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
        result = count-1
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
        return ratio

    def featureFourier(self, image):
        dft = cv2.dft(np.float32(image),flags = cv2.DFT_COMPLEX_OUTPUT)
        magnitude_spectrum = np.log(1+cv2.magnitude(dft[:,:,0],dft[:,:,1]))
        result = float(np.average(magnitude_spectrum))
        return result

    def featureRegressionRowAverages(self,image):
        height, width  = np.shape(image)
        verticalAverages = [0] * width
        for i in range(height-1):
            totalValue = 0
            for j in range(width):
                totalValue += image[i,j]
            verticalAverages[i] = int(totalValue/width)
        verticalModel = np.polyfit(range(len(verticalAverages)), verticalAverages, 1)   
        verticalResult = np.arctan(verticalModel[0])#
        return verticalResult

    def featureMoG(self, image):
        gmm, modelLabels, _ = self.mogModel
        image = (255-image)
        image = image.reshape(1,-1)
        prediction = modelLabels[int(gmm.predict(image))]
        return prediction

    def featureMeanBrightness(self, image):
        result = np.average(image)
        return result

    def featurePrototypeMatching(self, image):
        image = image.flatten()
        meanFeatureVector = np.zeros((10))
        for digit in range(10):
            meanFeatureVector[digit] = np.dot(image, self.meanImages[:, digit])
        return meanFeatureVector