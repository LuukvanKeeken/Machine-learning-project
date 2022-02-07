# Classes developed by the project group
from create_data import DataSets
from HCFeatures import HCFeatures

# functions
import numpy as np
from sklearn import linear_model

def featureVectorToDummyVariables(featureVector):
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

# Create instance of the dataset
createData = DataSets()
trainingData = createData.digits_standard()
testingData = createData.digits_testing()

# Create instance of handcrafted features
features = HCFeatures()
features.trainMeanImages(trainingData)
trainingRequired = features.trainingRequired
if trainingRequired:
    print("error no model for MoG")
    #features.trainMoG(trainingData)

# calculate feature vectors of training data for both visualization as linear regression on HC
trainingX, trainingY = trainingData
trainingFeature = np.zeros((1000,29))
for index, predictX in enumerate(trainingX):
    featureVector = features.predict(predictX)
    trainingFeature[index] = featureVectorToDummyVariables(featureVector)


# linear regression on the HC features
HCRegressionModel =  linear_model.LinearRegression()
HCRegressionModel.fit(trainingFeature, trainingY)

# testing on testing data
testingX, testingY  = testingData
missclassifiedDigits = np.zeros(10)
for index, predictX in enumerate(testingX):
    featureVector = featureVectorToDummyVariables(np.array(features.predict(predictX)))
    featureVector = featureVector.reshape(1, -1)
    HCpredictDigit = HCRegressionModel.predict(featureVector)
    if np.round(HCpredictDigit) != testingY[index]:
        missclassifiedDigits[testingY[index]] +=1
HCLoss = np.sum(missclassifiedDigits)/len(testingX)
print("Error of linear regression on features is " + str(HCLoss))