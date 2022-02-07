This repository contains the code that was used for a project for the course Machine Learning (2021/2022).

These are the main scripts of interest:
- ```modelLinearPixels.py```. Calculates linear regression weights on the pixel values.
- ```modelLinearHCFeatures.py```. Calculates the linear regression on the handcrafted feature vectors. Some features contain categorical data. With the function featureVectorToDummyVariables() dummy variables are created for the categories in order for linear regression to work.
- ```create_data.py```. Class which loads the dataset and splits it into training data and testing data. Class also supports augmentation of the data.
- ```HCFeatures.py```. Is the class with all handcrafted features.  By initializing the class it attempts loading the mixture of Gaussians model from the mogModel folder. The class supports training the prototype matching feature and the mixture of Gaussians model. With the function predict(predictX), the feature vector of a data point is calculated and returned.
- ```plottingAndTestingFeaturesAndRegression.py```. Was used to test features and create the plots for in the report.
- ```load_and_test_CNN.py```. Loads the saved CNN model and tests it on the test data. It shows the misclassifications and the confusion matrix.
- ```pipeline_cnn.py```. Main script used in searching for suitable hyperparameters for the CNN.