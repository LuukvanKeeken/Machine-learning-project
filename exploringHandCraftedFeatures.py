from re import I
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
import scipy.stats as stats
import math 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class handCrafted():

    def __init__(self):
        # Read in data
        data = []
        with open('mfeat-pix.txt') as f:
            lines = f.readlines()
            for line in lines:
                data_point = []
                for c in line:
                    if c.isnumeric():
                        data_point.append(int(c))
                data.append(data_point)
        data = np.asarray(data)

        # Split data in training and testing sets. For each class,
        # the first 100 vectors are for training, the other 100 vectors
        # are for testing.
        self.training_data = []
        self.test_data = []
        counter = 0
        for k in range(10):
            for i in range(2):
                for j in range(100):
                    if i == 0:
                        self.training_data.append(data[counter])
                    else:
                        self.test_data.append(data[counter])
                    counter += 1
        self.training_data = np.asarray(self.training_data)
        self.test_data = np.asarray(self.test_data)

        # Create arrays containing the labels that correspond to
        # the vectors in the training and testing sets.
        self.training_labels = []
        for i in range(10):
            self.training_labels += [i for j in range(100)]
        self.training_labels = np.asarray(self.training_labels)
        test_labels = self.training_labels
   

    def process(self):
        # TODO: standard mcp https://www.machinecurve.com/index.php/2019/07/27/how-to-create-a-basic-mlp-classifier-with-the-keras-sequential-api/
        emptyImage = np.zeros((16,15))
        featureVector = self.featureVector(emptyImage)
        samples = 100 # 100 for full training data set

        # number of input features
        numberX = len(featureVector)
        # number of hidden nodes
        numberH = numberX + 1
        # number of output nodes
        numberY = 10

        # create feature vectors of full dataset, with labels
        X = np.zeros((10*samples, numberX))
        y = np.zeros((10*samples, numberY))
        for j in range(samples):
            for i in range(numberY):
                index = samples*i + j
                imageX = np.reshape(self.training_data[index], (16,15))
                imageX *= int(255/imageX.max())
                imageX = imageX.astype(np.uint8)
                imageX = (255-imageX)
                inputX = self.featureVector(imageX)
                X[index] = inputX
                y[index][self.training_labels[index]] = 1


        # Train a very simple CNN on the training data, validate on the test data.
        training_data = np.zeros((800,numberX))
        training_labels = np.zeros((800,10))
        test_data = np.zeros((200,numberX))
        test_labels = np.zeros((200,10))
        testIndex = 0
        trainIndex = 0

        for j in range(100):
            for i in range(10):
                index = samples*i + j
                if j < 20:
                    test_data[testIndex] = X[index].copy()
                    test_labels[testIndex] = y[index].copy()
                    testIndex += 1
                else:
                    training_data[trainIndex] = X[index].copy()
                    training_labels[trainIndex] = y[index].copy()
                    trainIndex+=1
 
        model = models.Sequential()
        model.add(tf.keras.layers.Dense(300, input_shape = (numberX,), activation='relu'))
        model.add(tf.keras.layers.Dense(50, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        print(model.summary())

        model.compile(optimizer='adam',
                    loss='categorical_crossentropy', #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        history = model.fit(training_data, training_labels, epochs=10, batch_size = 800, validation_split = 0.2)

        test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)

        print(f"Error rate: {(1-test_acc)*100}%")
        print()



        #print(y[500])
        # random weights
        #W1 = np.random.rand(numberX, numberH)
        #W2 = np.random.rand(numberH, numberY)

        # loss = []
        # epochs = 10

        # for i in range(epochs):  
        #     m = len(X)
        #     output, A1 = self.forward(X, W1, W2)
        #     iter_loss=(1/(2*m))*np.sum((y-output)**2)
        #     loss.append(iter_loss)
        #     W1, W2 = self.backward(X, y, A1, output, W1, W2)

        # output, A1 = self.forward(X, W1, W2)
        # for i in range(len(output)):
        #     indexMax = np.argmax(output[i])
        #     for j in range(len(output[0])):
        #         if j == indexMax:
        #             output[i][j] = 1
        #         else:
        #             output[i][j] = 0



        # X = np.zeros((10*samples, numberX))
        # y = np.zeros((10*samples, numberY))
        # for j in range(samples):
        #     for i in range(numberY):
        #         index = samples*i + j
        #         imageX = np.reshape(self.training_data[index], (16,15))
        #         inputX = self.featureVector(imageX)
        #         output, A1 = self.forward(X, W1, W2)
        #         X[index] = inputX
        #         y[index][self.training_labels[index]] = 1
    
        featureResult = np.zeros((10,100))

        # Plot first digit for each class in training data.
        for j in range(100):
            for i in range(10):
                index = 100*i + j
                image = np.reshape(self.training_data[index], (16,15))
                image *= int(255/image.max())
                image = image.astype(np.uint8)
                image = (255-image)
                label = self.training_labels[index]
                featureResult[i,j] += self.feature(image.copy())
                featureVector = self.featureVector(image)
                print(featureVector)
            

        self.plotFeatureResult(featureResult)
        print()

    def featureVector(self, image):
        featureVector = []
        #featureVector.append(self.featureInk(image.copy()))
        featureVector.append(self.featureIslands(image.copy()))
        featureVector.append(self.featureTopCurve(image.copy()))
        featureVector.append(self.featureVerticalSymmetry(image.copy()))
        featureVector.append(self.featureHorizontalSymmetry(image.copy()))
        featureVector.append(self.featureLaplacian(image.copy()))
        featureVector.append(self.featureNonWhitePixels(image.copy()))
        return featureVector



    def featureInk(self, image):
        result = np.average(image)/255
        return result

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
        finaleImage = image.copy()
        
        threshold = 100
        image[image < threshold] = 0
        image[image > 0] = 1
        self.graph = image.copy()
        
        count = 0
        for i in range(len(image)):
            for j in range(len(image[0])):
                # If a cell with value 1 is not visited yet,
                # then new island found
                if self.graph[i][j] == 1:
                    # Visit all cells in this island
                    # and increment island count
                    self.DFS(i, j)
                    count += 1

        #finaleImage = image
        scale = 30
        width, height = np.shape(finaleImage)
        image = cv2.resize(finaleImage, (width*scale, height*scale), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow("test",image) #Original image
        #cv2.waitKey(0) 
        if count == 1:
            result = 0
        elif count == 2:
            result = 0.5
        else:
            result = 1
        return result

    def featureTopCurve(self,image):
        image[image >= 20] = 255
        image[image < 20] = 0
        width, height = np.shape(image)
        distance = []

        for i in range(width):
            for j in range(height):
                if image[i,j] < 20:
                    distance.append(j)
                    break            

        model = np.polyfit(range(len(distance)), distance, 2)#, full = True)
        lspace = np.linspace(0, 15, 16)#, 100)
        draw_x = lspace
        draw_y = np.polyval(model, draw_x)   # evaluate the polynomial
        
        modelTwo = np.polyfit(draw_x, draw_y, 1, full = True)
        residuals = modelTwo[1][0]
        result = residuals/280
        # draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)   # needs to be int32 and transposed
        # cv2.polylines(image, [draw_points], False, (100), thickness = 1)  # args: image, points, closed, color
        # scale = 30
        # image = cv2.resize(image, (15*scale, 16*scale))
        # cv2.imshow("test",image) #Original image
        # cv2.waitKey(0) 
        return result

    def featureVerticalSymmetry(self, image):
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
        return pixelsTop / total

    def featureHorizontalSymmetry(self, image):
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
        return pixelsLeft / total

    def featureLaplacian(self, image):
        # add extra boarder to image
        width, height = np.shape(image)
        column = 255 * np.ones(height)
        row = 255 * np.ones((width+2,1))
        image = np.vstack((column,image))
        image = np.vstack((image,column))
        #width, _ = np.shape(image)
        
        image = np.hstack((image, row))
        image = np.hstack((row,image))
        image = image.astype(np.uint8)

        # laplacian filter is sensitive to noise because it calculates the 2nd derivative. First smooth image.
        kernelSize = (5,5)
        gaussian = cv2.GaussianBlur(image,kernelSize,1)

        

        # frequency domain
        #Create a 2 dimensional array (3x3) and fill the array and apply the array as 2Dfilter.
        #use -1 to use the same dept as the original image
        CustomLaplacianImg = cv2.filter2D(gaussian,-1,np.array([[1,1,1], [1,-8,1],[1,1,1]]))
        # Kernel with bad results:
        #CustomLaplacianImg = cv2.filter2D(gaussian,-1,np.array([[0,1,0], [1,-4,1],[0,1,0]]))
        # Standard laplacian filter and scale back to get rid of the visualisation image
        StandardLaplacianImg = cv2.convertScaleAbs(cv2.Laplacian(gaussian,cv2.CV_16S, 3))

        laplacianPixels = np.sum(StandardLaplacianImg)
        imageInk = np.sum(255-image)
        ratio = laplacianPixels/imageInk

        #finaleImage = StandardLaplacianImg
        #scale = 30
        #width, height = np.shape(finaleImage)
        #image = cv2.resize(finaleImage, (width*scale, height*scale))
        #cv2.imshow("test",image) #Original image
        #cv2.waitKey(0) 
        return ratio

    def featureNonWhitePixels(self, image):       
        threshold = np.max(image)
        image[image < threshold] = 0
        image[image > 0] = 1
        totalPixels = len(image)*len(image[0])
        nonWhite = np.count_nonzero(image)
        ratio = nonWhite/totalPixels

        #finaleImage = image
        #scale = 30
        #width, height = np.shape(finaleImage)
        #image = cv2.resize(finaleImage, (width*scale, height*scale), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow("test",image) #Original image
        #cv2.waitKey(0) 

        result = ratio
        return result 

    def feature(self, image):
        # add extra boarder to image
        width, height = np.shape(image)
        column = 255 * np.ones(height)
        row = 255 * np.ones((width+2,1))
        #image = np.vstack((column,image))
        #image = np.vstack((image,column))       
        #image = np.hstack((image, row))
        #image = np.hstack((row,image))
        #image = image.astype(np.uint8)
        #finaleImage = image.copy()
        
        threshold = np.max(image)
        image[image < threshold] = 0
        image[image > 0] = 1
        totalPixels = len(image)*len(image[0])
        nonWhite = np.count_nonzero(image)
        ratio = nonWhite/totalPixels


        finaleImage = image
        scale = 30
        width, height = np.shape(finaleImage)
        image = cv2.resize(finaleImage, (width*scale, height*scale), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow("test",image) #Original image
        #cv2.waitKey(0) 

        result = ratio
        return result

  
    # A utility function to do DFS for a 2D
    # boolean matrix. It only considers
    # the 8 neighbours as adjacent vertices
    def DFS(self, i, j):
        if i < 0 or i >= len(self.graph) or j < 0 or j >= len(self.graph[0]) or self.graph[i][j] != 1:
            return
  
        # mark it as visited
        self.graph[i][j] = -1
  
        # Recur for 8 neighbours
        self.DFS(i - 1, j - 1)
        self.DFS(i - 1, j)
        self.DFS(i - 1, j + 1)
        self.DFS(i, j - 1)
        self.DFS(i, j + 1)
        self.DFS(i + 1, j - 1)
        self.DFS(i + 1, j)
        self.DFS(i + 1, j + 1)
  
    # The main function that returns
    # count of islands in a given boolean
    # 2D matrix

        # image[image >= 20] = 255
        # image[image < 20] = 0
        # width, height = np.shape(image)
        # distance = []

        # pixelsLeft = 0
        # pixelsRight = 0
        # pixelsTop = 0
        # pixelsBottom = 0

        # for i in range(width):
        #     for j in range(height):
        #         if image[i,j] == 0:
        #             if i < width/2:
        #                 pixelsLeft += 1
        #             else:
        #                 pixelsRight += 1
        #             if j < height/2:
        #                 pixelsTop += 1
        #             else:
        #                 pixelsBottom += 1
        # total = pixelsLeft + pixelsRight
        # symmetry = pixelsTop / total
        # result = symmetry




        # model = np.polyfit(range(len(distance)), distance, 2)#, full = True)
        # lspace = np.linspace(0, 15, 16)#, 100)
        # draw_x = lspace
        # draw_y = np.polyval(model, draw_x)   # evaluate the polynomial
        # modelTwo = np.polyfit(draw_x, draw_y, 1, full = True)
        # result = modelTwo[1]/280
        # draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)   # needs to be int32 and transposed
        
        # cv2.polylines(image, [draw_points], False, (100), thickness = 1)  # args: image, points, closed, color
        # scale = 30
        # image = cv2.resize(image, (15*scale, 16*scale))
        # cv2.imshow("test",image) #Original image
        # cv2.waitKey(0) 
    
        #shape = image.shape()

        #np.average(image)/255



        #vis2 = cv2.cvtColor(image, cv2.COLOR_GR)
        #image[image >= 6] = 255
        #image[image < 6] = 0
        #image = image.astype(np.uint8)
        #
        #img = (255-img)
        #image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]
        
        #cv2.cvt

        # SobelImgX = cv2.convertScaleAbs(cv2.Sobel(image,cv2.CV_16S,1,0,ksize=3))
        # SobelImgY = cv2.convertScaleAbs(cv2.Sobel(image,cv2.CV_16S,0,1,ksize=3))
        # SobelImgXY = cv2.addWeighted(SobelImgX, 0.5,SobelImgY,0.5,0)


        # #ax = sns.heatmap(image, cmap='gray_r')
        # #plt.show()
        # plt.subplots(figsize=(20, 14)) #Define size of the subplots
        # PltOriginalImg = plt.subplot(2,2,1)#Place 1
        # PltOriginalImg.axis('off') #disable axis
        # PltSobelX = plt.subplot(2,2,2)#Place 2
        # PltSobelX.axis('off') #disable axis
        # PltSobelY = plt.subplot(2,2,3)#Place 3
        # PltSobelY.axis('off') #disable axis
        # PltSobelXY = plt.subplot(2,2,4)#Place 4
        # PltSobelXY.axis('off') #disable axis
        # 1
        # #Plot all images
        # PltOriginalImg.imshow(image,cmap = 'gray') #Original image
        # PltSobelX.imshow(SobelImgX,cmap = 'gray') #Sobel X direction image
        # PltSobelY.imshow(SobelImgY,cmap = 'gray') #Sobel Y direction image
        # PltSobelXY.imshow(SobelImgXY,cmap = 'gray') #Sobel XY direction image
        # plt.show() #show images

        # if testlabel == 1:
        #     result = 0.9
        # else:
        #     result = random.random()

        

    def plotFeatureResult(self,featureResult):
        #x = np.linspace(-0.4, 1.5, 100)
        
        print(np.min(featureResult))
        print(np.max(featureResult))
        bins = 50
        x = np.linspace(0.0, 1.0, bins)
        for i in range(10):
            #mu = np.average(featureResult[i])
            #sigma = np.std(featureResult[i])
            #plt.plot(x, stats.norm.pdf(x, mu, sigma), label = str(i))
            
            digit =  featureResult[i]
            values = np.histogram(digit, bins, (0.0, 1.0))[0]
            plt.plot(x, values, label = str(i))

        plt.legend()
        plt.show()
        print()




if __name__ == "__main__":
    program = handCrafted()
    program.process()

# # Train a very simple CNN on the training data, validate on the test data.
# training_data, test_data = training_data/6.0, test_data/6.0
# training_data = np.reshape(training_data, (1000, 16, 15))
# test_data = np.reshape(test_data, (1000, 16, 15))

# model = models.Sequential()
# model.add(layers.Conv2D(31, (2, 2), activation='relu', input_shape=(16, 15, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(62, (2, 2), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(62, (2, 2), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(62, activation='relu'))
# model.add(layers.Dense(10))
# print(model.summary())

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# history = model.fit(training_data, training_labels, epochs=10, 
#                     validation_data=(test_data, test_labels))

# test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)

# print(f"Error rate: {(1-test_acc)*100}%")
