from create_data import DataSets
from sklearn import linear_model
import numpy as np

# Create instance of the dataset
createData = DataSets()
trainingData = createData.digits_standard()
testingData = createData.digits_testing()

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
#self.plotClassificationResult(missclassifiedDigits, len(testingX), "Error linear regression on pixels", "classificationLinearRegression.png")