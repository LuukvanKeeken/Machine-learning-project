# other classes developed by the project group
from create_data import DataSets
from HCFeatures import HCFeatures

# functions
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from matplotlib.ticker import MaxNLocator
import string
from sklearn import linear_model

class featurePipeline():

    def __init__(self):
        pass

    def plotExperimentResult(self,featuresResult):
        experimentResults = []
        experimentResults.append(["Vertical ratio k=3",featuresResult[0]])
        experimentResults.append(["Vertical ratio k=8",featuresResult[1]])
        experimentResults.append(["Islands",featuresResult[2]])
        experimentResults.append(["Laplacian",featuresResult[3]])
        experimentResults.append(["Fourier",featuresResult[4]])
        experimentResults.append(["Regression on row averages",featuresResult[5]])
        experimentResults.append(["Mixture of Gaussians",featuresResult[6]])
        experimentResults.append(["Mean brightness",featuresResult[7]])

        plots = len(experimentResults)
        int(plots/2)
        plotsHorizontal = 4
        plotsVertical = math.ceil(plots/plotsHorizontal)# + plots - int(plots/plotsHorizontal)*plotsHorizontal
        fig, ax = plt.subplots(plotsVertical, plotsHorizontal, figsize=(8, 8))#, subplot_kw=dict(xticks=[], yticks=[]))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        fig.set_size_inches(15, 10)      
        maxY = 0
        for i in range(plots):
            axi = ax.flat[i]
            experimentName = experimentResults[i][0]
            featureResult = experimentResults[i][1]
            maxValue = np.max(featureResult)
            minValue = np.min(featureResult)
            axi.set_title(experimentName)
            axi.set_xlabel('value')
            axi.set_ylabel('counts')
            axi.set_xlim(minValue, maxValue)
            axi.yaxis.set_major_locator(MaxNLocator(integer=True))
            axi.text(-0.1, 1.1, string.ascii_uppercase[i], transform=axi.transAxes, size=20, weight='bold')
            bins = 50
            x = np.linspace(minValue, maxValue, bins)           
            for i in range(10):
                digit =  featureResult[i]
                values = np.histogram(digit, bins, (minValue, maxValue))[0]
                if np.max(values) > maxY:
                    maxY = np.max(values)  
                axi.plot(x, values, label = str(i))                 
                handles, labels = axi.get_legend_handles_labels()

        plt.legend(handles = handles, labels = labels, loc='upper center', 
             bbox_to_anchor=(-1.5, -0.2),fancybox=False, shadow=False,
             ncol=10)
        plt.savefig("HCfeatures.png", dpi = 300, bbox_inches='tight') # when saving, specify the DPI
  
    def plotFourier(self, image):
        image = np.reshape(image, (16,15))
        image = 6-image
        row = image[13]
        fig, ax = plt.subplots(2, 3, figsize=(8, 5.5))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        for i, axi in enumerate(ax.flat):
            axi.text(-0.1, 1.1, string.ascii_uppercase[i], transform=axi.transAxes, size=20, weight='bold')
        ax.flat[0].imshow(image,cmap = 'gray')
        ax.flat[1].plot(row)
        x = np.linspace(0, len(row)-1, len(row))
        xnew = np.linspace(0, len(row)-1, 100)
        dft = cv2.dft(np.float32(row),flags = cv2.DFT_COMPLEX_OUTPUT)
        ax.flat[2].plot(dft[:,:,0]) # at 2 and 13 are peeks of (7, 7) and (7,-7)
        ax.flat[2].plot(x, [7.1]*15, '--')
        base = 2*math.pi/15 * (xnew+1)
        scale = 2.29/(len(row))
        ax.flat[3].plot(xnew, np.cos(base*2)*7*scale)
        ax.flat[3].plot(xnew, np.cos(base*13)*scale*7*2/13)
        ax.flat[3].plot(x, [0.8]*15)
        ax.flat[4].plot(xnew, (np.cos(base*2)*7 + np.cos(base*13)*7*2/13)*scale+1)
        ax.flat[5].plot(row)
        ax.flat[5].plot(xnew, (np.cos(base*2)*7 + np.cos(base*13)*7*2/13)*scale+1)
        plt.savefig("Fourier.png", dpi = 300, bbox_inches='tight')
        print()

    def plotClassificationResult(self, wrongDigits, totalDigits, title, saveName):
        fig, ax = plt.subplots()
        ax.set_ylim(0,100)
        ax.set_xlim(-0.5,9.5)
        plt.bar(range(len(wrongDigits)), wrongDigits)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.set_size_inches(8, 6)
        averageError = 100*np.sum(wrongDigits)/totalDigits
        x = np.linspace(-1, len(wrongDigits), len(wrongDigits)+1)
        plt.plot(x, [averageError]*len(x), '--', color = 'C1')
        plt.xticks(np.arange(0, len(wrongDigits), 1))
        plt.title(title)
        plt.xlabel('Digit')
        plt.ylabel('Error percentage')
        plt.savefig(saveName, dpi = 300, bbox_inches='tight')

    def featureVectorToDummyVariables(self, featureVector):
        # island feature = 2
        if featureVector[2] == 0.5:
            featureVector = np.append(featureVector, 1)
        else:
            featureVector = np.append(featureVector, 0)
        if featureVector[2] == 1.0:
            featureVector = np.append(featureVector, 1)
        else:
            featureVector = np.append(featureVector, 0)
        if featureVector[2] == 0.0:
            featureVector[2] = 1   
        else:
            featureVector[2] = 0

        # MoG feature = 6
        prediction = featureVector[6]
        appendVector = np.zeros(9)
        if prediction == 0:
            featureVector[6] = 1
        else:
            featureVector[6] = 0
            appendVector[int(prediction-1)] = 1
        featureVector = np.append(featureVector, appendVector)
        return featureVector

    def linearRegressionPixels(self, trainingData, testingData):
        # training on training data
        trainingX, trainingY = trainingData
        pixelRegressionModel = linear_model.LinearRegression() 
        pixelRegressionModel.fit(trainingX, trainingY)

        # testing on testing data
        testingX, testingY  = testingData
        missclassifiedDigits = np.zeros(10)
        for index, predictX in enumerate(testingX):
            prediction = pixelRegressionModel.predict(predictX.reshape(1,-1))
            if np.round(prediction) != testingY[index]:
                missclassifiedDigits[testingY[index]] +=1
        
        error = np.sum(missclassifiedDigits)/len(testingX)
        print("Error of linear regression on pixels is " + str(error))
        self.plotClassificationResult(missclassifiedDigits, len(testingX), "Error linear regression on pixels", "classificationLinearRegression.png")

    def linearRegressionHC(self, features, trainingData, testingData):
        # training on training features
        trainingFeature, trainingY = trainingData
        HCRegressionModel =  linear_model.LinearRegression()
        HCRegressionModel.fit(trainingFeature, trainingY)
        
        # testing on testing data
        testingX, testingY  = testingData
        missclassifiedDigits = np.zeros(10)
        for index, predictX in enumerate(testingX):
            featureVector = self.featureVectorToDummyVariables(np.array(features.predict(predictX)))
            featureVector = featureVector.reshape(1, -1)
            HCpredictDigit = HCRegressionModel.predict(featureVector)
            if np.round(HCpredictDigit) != testingY[index]:
                missclassifiedDigits[testingY[index]] +=1
        HCLoss = np.sum(missclassifiedDigits)/len(testingX)
        print("Error of linear regression on features is " + str(HCLoss))
        self.plotClassificationResult(missclassifiedDigits, len(testingX), "Error linear regression on HC features", "classificationFeatureRegression.png")       

    def pipeline(self):
        # Create instance of the dataset
        createData = DataSets()
        trainingData = createData.digits_standard()
        testingData = createData.digits_testing()
        
        # Create the example plot of the Fourier transformation for report
        self.plotFourier(trainingData[0][640])

        # Create instance of handcrafted features
        features = HCFeatures()
        features.trainMeanImages(trainingData)
        trainingRequired = features.trainingRequired
        if trainingRequired:
            print("error no model for MoG")
            #features.trainMoG(trainingData)

        # linear regression on pixels
        self.linearRegressionPixels(trainingData, testingData)

        # calculate feature vectors of training data for both visualization as linear regression on HC
        trainingX, _ = trainingData
        regressionX = np.zeros((1000,29))
        regressionY = np.zeros((1000))
        featuresResult = np.zeros((18, 10,100))
        for index, predictX in enumerate(trainingX):
            digit = int(index/100)
            number = index - digit*100
            featureVector = features.predict(predictX)
            featuresResult[:, digit, number] = featureVector
            regressionX[index] = self.featureVectorToDummyVariables(featureVector)
            regressionY[index] = digit

        # linear regression on the HC features
        featureTrainingData = regressionX, regressionY
        self.linearRegressionHC(features, featureTrainingData, testingData)

        # plot the feature results of the training dataset
        self.plotExperimentResult(featuresResult)     
    
if __name__ == "__main__":
    program = featurePipeline()
    program.pipeline()