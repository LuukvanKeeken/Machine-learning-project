from re import I
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
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
    
    def plot_digits(self, data):
        fig, ax = plt.subplots(10, 10, figsize=(8, 8), subplot_kw=dict(xticks=[], yticks=[]))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        for i, axi in enumerate(ax.flat):
            im = axi.imshow(data[i].reshape(16, 15), cmap='binary')
            #im.set_clim(0, 16)
        plt.show()

    def process(self):
        # TODO: standard mcp https://www.machinecurve.com/index.php/2019/07/27/how-to-create-a-basic-mlp-classifier-with-the-keras-sequential-api/
        emptyImage = np.zeros((16,15))
        #featureVector = self.featureVector(emptyImage)
        samples = 100 # 100 for full training data set

        numberY = 10

        featureResult = np.zeros((10,100))

        # dimension reduction
        print(self.training_data.shape)


        scaledData = self.training_data.copy()
        scaledData *= int(255/scaledData.max())
        scaledData = scaledData.astype(np.uint8)
        #scaledData = (255-scaledData)
        data = scaledData
        #pca = PCA(0.99, whiten=True)
        #data = pca.fit_transform(self.training_data)
        #data = self.training_data
        #print(data.shape)

        #self.plot_digits(data[0:100])

        # MoG per class.
        parameters = []
        models = []
        for i in range(10):
            start = i*100
            end = start + 100
            classDataset = data[start:end]
            #n_components = np.arange(1, 6, 1)
            #models = [GaussianMixture(n, covariance_type='full', random_state=0) for n in n_components]
            #aics = [model.fit(classDataset).aic(classDataset) for model in models]
            #plt.plot(n_components, aics)
            #plt.show()
            # Per class, 2 gaussians has lowes AIC
            gmm = GaussianMixture(2, covariance_type='full', random_state=0)
            gmm.fit(classDataset) 
            models.append(gmm.copy())
            #data_new = gmm.sample(100)[0]
            #print(data_new.shape)
            #digits_new = pca.inverse_transform(data_new)
            #digits_new = data_new
            #self.plot_digits(digits_new)
            #print()
        
        for i in range(10):
            testIndex = random.randint(0,1000)
            image = data[testIndex]
            realLabel = int(testIndex/100)
            predictions = []
            for model in range(len(models)):
                image = image.reshape(1,-1)
                prediction = models[model].predict_proba(image)
                predictions.append(prediction)
                print()
            print()
        print()



        

        # MoG for all classes combined.
        #n_components = np.arange(10, 40, 2)
        #models = [GaussianMixture(n, covariance_type='full', random_state=0) for n in n_components]
        #aics = [model.fit(data).aic(data) for model in models]
        #plt.plot(n_components, aics)
        #plt.show() 
        # without PCS 20 components looks best
        # With 99% PCA ... components looks best to minimize the AIC

        gmm = GaussianMixture(20, covariance_type='full', random_state=0)
        gmm.fit(data) 
        data_new = gmm.sample(100)[0]
        print(data_new.shape)
        #digits_new = pca.inverse_transform(data_new)
        digits_new = data_new
        self.plot_digits(digits_new)




                
            

        
        # Plot first digit for each class in training data.
        for j in range(100):
            for i in range(10):
                index = 100*i + j
                image = np.reshape(self.training_data[index], (16,15))
                image *= int(255/image.max())
                image = image.astype(np.uint8)
                image = (255-image)
                featureResult[i,j] += self.feature(image.copy())
                featureVector = self.featureVector(image.copy())

        self.plotFeatureResult(featureResult)
        print()




 

    def featureVector(self, image):
        featureVector = []
        featureVector.append(self.featureInk(image.copy()))
        featureVector.append(self.featureIslands(image.copy()))
        featureOne, featureTwo = self.featureTopCurve(image.copy())
        featureVector.append(featureOne)
        featureVector.append(featureTwo)
        featureOne, featureTwo = self.featureSymmetry(image.copy())
        featureVector.append(featureOne)
        featureVector.append(featureTwo)
        #featureVector.append(self.featureSymmetry(image.copy()))
        #featureVector.append(self.featureHorizontalSymmetry(image.copy()))
        featureVector.append(self.featureLaplacian(image.copy()))
        featureVector.append(self.featureNonWhitePixels(image.copy()))
        featureVector.append(self.featureSingeLineNonWhite(image.copy()))
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
                    islandLocation = (i, j)

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
            islandLocation
        else:
            result = 1
        return result

    def featureTopCurve(self,image):
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

        # draw a new linear space because the line does not start at 0.
        # draw_x = np.linspace(firstPixel, firstPixel + len(distance) -1, len(distance))
        # draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)   # needs to be int32 and transposed
        # cv2.polylines(image, [draw_points], False, (100), thickness = 1)  # args: image, points, closed, color   
        # drawLinear =  np.polyfit(lspace, draw_y, 1)
        # draw_y = np.polyval(drawLinear, lspace)   # evaluate the polynomial
        # draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)   # needs to be int32 and transposed
        # cv2.polylines(image, [draw_points], False, (100), thickness = 1)  # args: image, points, closed, color
        
        #scale = 30
        #width, height = np.shape(image)
        #largeImage = cv2.resize(image, (width*scale, height*scale), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow("test",largeImage) #Original image
        #cv2.waitKey(0) 
        return result, angle

    def featureSymmetry(self, image):
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
        return horizontalSymmetry, verticalSymmetry

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


        #kernel = np.array([[0,0,1], [0,-3,1],[0,0,1]])
        #ratio -= 0.17
        #ratio *= 2.4
        
        #kernel = np.array([[0,0,0], [1,-2,1],[0,0,0]])
        #ratio -= 0.05
        #ratio *= 9

        # kernel = np.array([[1, 0, 1],
        #                    [0,-4, 0],
        #                    [1, 0, 1]])

        # kernel = np.array([[1, 0, 1],
        #                    [3,-10, 3],
        #                    [1, 0, 1]])


        # laplacianImgX = cv2.filter2D(gaussian,-1,kernel)
        
        # laplacianPixels = np.sum(laplacianImgX)
        #ratio = laplacianPixels/imageInk
        #ratio -= 0.05
        #ratio *= 9
        #ratio += 0.3

        finaleImage = StandardLaplacianImg
        scale = 30
        width, height = np.shape(finaleImage)
        largeImage = cv2.resize(finaleImage, (width*scale, height*scale), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow("test",largeImage) #Original image
        #cv2.waitKey(0) 
        return ratio

    def featureNonWhitePixels(self, image):       
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
        return ratio 

    def featureSingeLineNonWhite(self,image):
        lineImage = cv2.ximgproc.thinning(255-image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)     
        totalPixels = len(lineImage)*len(lineImage[0])
        nonWhite = np.count_nonzero(lineImage)
        ratio = nonWhite/totalPixels
        ratio -= 0.08
        ratio *= 4.898
        return ratio

    def feature(self, image):
        lineImage = cv2.ximgproc.thinning(255-image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)   
        scale = 30
        width, height = np.shape(lineImage)
        largeImage = cv2.resize(lineImage, (width*scale, height*scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("test",largeImage) #Original image
        cv2.waitKey(0)

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
                    self.DFS(i, j, count+2)
                    count += 1
                    islandTop = i

        #finaleImage = image
        scale = 30
        width, height = np.shape(finaleImage)
        largeImage = cv2.resize(finaleImage, (width*scale, height*scale), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow("test",largeImage) #Original image
        #cv2.waitKey(0)
        if count == 1:
            result = 0
        elif count == 2:
            
            for i in range(len(image)):
                for j in range(len(image[0])):
                    if self.graph[i][j] == 3:
                        islandBottom = i
            islandCenter = islandTop + (islandBottom - islandTop)/2
            
            distance = []
            distance.append(abs(len(image)/4 - islandCenter))
            distance.append(abs(len(image)/2 - islandCenter))
            distance.append(abs((len(image)*3)/4 - islandCenter))
            result = 0.2 + np.argmin(distance) * 0.2

        else:
            result = 1 #island 3 and 4
        return result
       
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


        #kernel = np.array([[0,0,1], [0,-3,1],[0,0,1]])
        #ratio -= 0.17
        #ratio *= 2.4
        
        #kernel = np.array([[0,0,0], [1,-2,1],[0,0,0]])
        #ratio -= 0.05
        #ratio *= 9

        # kernel = np.array([[1, 0, 1],
        #                    [0,-4, 0],
        #                    [1, 0, 1]])

        # kernel = np.array([[1, 0, 1],
        #                    [3,-10, 3],
        #                    [1, 0, 1]])


        # laplacianImgX = cv2.filter2D(gaussian,-1,kernel)
        
        # laplacianPixels = np.sum(laplacianImgX)
        #ratio = laplacianPixels/imageInk
        #ratio -= 0.05
        #ratio *= 9
        #ratio += 0.3
        
        

        finaleImage = image
        scale = 30
        width, height = np.shape(finaleImage)
        largeImage = cv2.resize(finaleImage, (width*scale, height*scale), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow("test",largeImage) #Original image
        #cv2.waitKey(0) 
        # pip install opencv-contrib-python
        kernelSize = (9,9)
        
        threshold = 100
        largeImage[largeImage < threshold] = 0
        largeImage[largeImage > 0] = 255
        #largeImage = cv2.GaussianBlur(largeImage,kernelSize,1)
        #largeImage = cv2.GaussianBlur(largeImage,kernelSize,1)
        #largeImage = cv2.GaussianBlur(largeImage,kernelSize,1)
        #lineImage = cv2.ximgproc.thinning(255-largeImage, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        lineImage = cv2.ximgproc.thinning(255-image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        scale = 30
        width, height = np.shape(lineImage)
        largeLineImage = cv2.resize(lineImage, (width*scale, height*scale), interpolation=cv2.INTER_NEAREST)

     
        totalPixels = len(lineImage)*len(lineImage[0])
        nonWhite = np.count_nonzero(lineImage)
        ratio = nonWhite/totalPixels
        ratio *= 5.884
        #ratio -= 0.4
        #ratio = min(1,ratio)
        #ratio = max(0,ratio)
        #ratio 

        islands = self.featureIslands(image)
        if islands < 0.4:
            allowedIntersections = 0
        elif islands >= 0.4 and islands <= 0.6:
            allowedIntersections = 1
        else:
            allowedIntersections = 2
        
        process = True
        #while process:
        #    self.getInersections(largeImage)


        # if no islands, the line should be one piece

        
        change = False
        while change:
            threshold = 100
            tempImage = largeImage.copy()
            tempImage[tempImage < threshold] = 0
            tempImage[tempImage > 0] = 255

            kernelSize = (9,9)
            tempImage = cv2.GaussianBlur(tempImage,kernelSize,1)
            if True: #self.featureIslands(tempImage) == orignalIslands:
                largeImage = tempImage.copy()
                cv2.imshow("test",largeImage) #Original image
                cv2.waitKey(0)
            else:
                change = False


        #largeImage = cv2.addWeighted()


        #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        #largeImage = cv2.filter2D(largeImage, -1, kernel)
        #final_frame = cv2.hconcat((largeImage, largeLineImage))
        #cv2.imshow("test",final_frame) #Original image
        #cv2.waitKey(0) 

        # idea: if there is a really large black circle, there must be a white spot in the middel. Add it manually?

        return ratio


    # A utility function to do DFS for a 2D
    # boolean matrix. It only considers
    # the 8 neighbours as adjacent vertices
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
  
    def trainNetwork(self, numberX, numberY, samples):
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
