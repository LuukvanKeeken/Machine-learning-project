from re import I, L
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import math
from create_data import DataSets
from HCFeatures import HCFeatures

class featurePipeline():

    def __init__(self):
        self.mogModel = None
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

    def plotExperimentResult(self,experimentResult):
        plots = len(experimentResult)
        int(plots/2)
        plotsHorizontal = 4
        plotsVertical = math.ceil(plots/plotsHorizontal)# + plots - int(plots/plotsHorizontal)*plotsHorizontal
        fig, ax = plt.subplots(plotsVertical, plotsHorizontal, figsize=(8, 8))#, subplot_kw=dict(xticks=[], yticks=[]))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        fig.set_size_inches(15, 10)
        
        maxY = 0
        for i in range(plots):
            axi = ax.flat[i]
        #for i, axi in enumerate(ax.flat):
            experimentName = experimentResult[i][0]
            featureResult = experimentResult[i][1]
            maxValue = np.max(featureResult)
            axi.set_title(experimentName)
            axi.set_xlabel('value')
            axi.set_ylabel('counts')
            axi.set_xlim(0, maxValue)

            
            #for featureResult in experimentResult: 
            # always start at 0.
            

            bins = 50
            x = np.linspace(0.0, maxValue, bins)
            
            for i in range(10):
                digit =  featureResult[i]
                values = np.histogram(digit, bins, (0.0, maxValue))[0]
                if np.max(values) > maxY:
                    maxY = np.max(values)
                axi.plot(x, values, label = str(i))
                axi.legend()
            
                #axi.grid(True)
            
        #for i, axi in enumerate(ax.flat):
        #    axi.set_ylim(0, maxY)
        # for i in [2, 3]:
        #     plt.delaxes(ax.flatten()[i])
        #plt.legend()
        plt.tick_params(axis='both', which='major', labelsize=5)
        
        plt.savefig("HCfeatures.png", dpi = 300, bbox_inches='tight') # when saving, specify the DPI
        #plt.show()
        
        print() 

    def pipeline(self):
        experimentResults = []
        #self.trainMoG()
        if False:
            for experiment in range(2):
                featureResult = np.zeros((10,100))
                value = experiment
                for j in range(100):
                    #for i in range(10):
                    for i in (2,3,5):
                        index = 100*i + j
                        image = np.reshape(self.training_data[index], (16,15))
                        image *= int(255/image.max())
                        image = image.astype(np.uint8)
                        image = (255-image)
                        featureResult[i,j] += self.featureTopCurve(image.copy())#, value)
                experimentResults.append([str(value),featureResult])
            self.plotExperimentResult(experimentResults)
        
        if True:
            createData = DataSets()
            trainingData = createData.digits_standard()
   
            features = HCFeatures()
            features.fit(trainingData)

            testingX, testingY = createData.digits_testing()

            for _, predictX in enumerate(testingX):
                featureVector = features.predict(predictX)
                print()

            featuresResult = np.zeros((7, 10,100))
            for j in range(100):
                for i in range(10):
                    index = 100*i + j
                    image = np.reshape(self.training_data[index], (16,15))
                    image *= int(255/image.max())
                    image = image.astype(np.uint8)
                    image = (255-image)
                    featuresResult[0, i,j] += self.featureHorizontalSymmetry(image.copy(), xParameter = 3)
                    featuresResult[1, i,j] += self.featureHorizontalSymmetry(image.copy(), xParameter = 8)
                    featuresResult[2, i,j] += self.featureIslands(image.copy()) # No parameter required
                    featuresResult[3, i,j] += self.featureLaplacian(image.copy()) # No parameter required
                    featuresResult[4, i,j] += self.featureFourier(image.copy()) # looks random, but may be usefull
                    featuresResult[5, i,j] += self.featureVerticalPolyRow(image.copy())
                    featuresResult[6, i,j] += self.featureMoG(image.copy())

            experimentResults.append(["horizontal symmetry x=3",featuresResult[0]])
            experimentResults.append(["horizontal symmetry x=8",featuresResult[1]])
            experimentResults.append(["islands",featuresResult[2]])
            experimentResults.append(["Laplacian",featuresResult[3]])
            experimentResults.append(["Fourier",featuresResult[4]])
            experimentResults.append(["verticalPolyRow",featuresResult[5]])
            experimentResults.append(["MoG",featuresResult[6]])
            self.plotExperimentResult(experimentResults)
        
        if False:
            self.trainMoG()
            good = 0
            for j in range(100):
                for i in range(10):
                    index = 100*i + j
                    image = np.reshape(self.training_data[index], (16,15))
                    image *= int(255/image.max())
                    image = image.astype(np.uint8)
                    image = (255-image)
                    classification = self.HCtree(image)
                    if classification == i:
                        good +=1
            print(str(good/1000))
                    
        print()

    def HCtree(self, image):
        feature0 = self.featureHorizontalSymmetry(image.copy(), xParameter = 3)
        feature1 = self.featureHorizontalSymmetry(image.copy(), xParameter = 8)
        feature2 = self.featureIslands(image.copy()) # No parameter required
        feature3 = self.featureLaplacian(image.copy()) # No parameter required
        feature4 = self.featureFourier(image.copy()) # looks random, but may be usefull
        feature5 = self.featureVerticalPolyRow(image.copy())
        
        if feature2 > 0.8:
            # 8
            digit = 8
        elif feature2 < 0.2:
            # 1, 2, 3, 4, 5, 7
            if feature0 > 0.33:
                digit = 7
            else:
                # digit 1,2,3,4,5
                if feature3 < 0.3:
                    digit = 1
                else:
                    # digit 2,3,4,5
                    if feature1 < 0.45:
                        digit = 4
                    else:
                        # digit 2, 3, 5
                        digit = self.featureMoG(image)
                        print()          
        else:
            # 0, 6, 9
            if feature1 < 0.45:
                digit = 6
            elif feature1 > 0.59:
                digit = 9
            else:
                digit = 0


        return digit

    def featureMoG(self, image):
        gmm, modelLabels = self.mogModel
        image = (255-image)
        image = image.reshape(1,-1)
        prediction = modelLabels[int(gmm.predict(image))]
        return prediction/10

    def trainMoG(self):
        scaledData = self.training_data.copy()
        scaledData *= int(255/scaledData.max())
        scaledData = scaledData.astype(np.uint8)
        data = scaledData
        components = 20
        gmm = GaussianMixture(n_components=components,
                                covariance_type='full', 
                                tol=1e-10, # only effect when 0
                                reg_covar=1e-10, #default: 1e-06 
                                max_iter=100, 
                                n_init=20, # higher is better change on good model
                                init_params='kmeans', 
                                weights_init=None, 
                                means_init=None, 
                                precisions_init=None, 
                                random_state=None, 
                                warm_start=False, 
                                verbose=1,
                                verbose_interval=10)
        gmm.fit(data) 
        
        # find labels by the model
        labels = np.zeros((components, 10))
        for i in range(1000):
            realLabel = self.training_labels[i]
            image = data[i]
            image = image.reshape(1,-1)
            classPredictions = int(gmm.predict(image))
            labels[classPredictions][realLabel] += 1
        modelLabels = [0] * components
        for index in range(components):
            mostCount = np.argmax(labels[index])
            modelLabels[index] = mostCount
        self.mogModel = gmm, modelLabels
        return


    def featureTopCurve(self,image):
        #image[image >= 20] = 255
        image[image < 100] = 0
        height, width = np.shape(image)
        distance = [] 
        newImage = np.zeros((height, width))

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

        for i in range(width):
            if distance[i] < 50:
                newImage[distance[i],i] = 255

        
        plt.subplot(121),plt.imshow(image, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(newImage, cmap = 'gray')
        plt.title('New image'), plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.waitKey(0)

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
        angle = model[0]
        #angle += 1
        #angle /= 2
        residuals = linearModel[1][0]
        points = len(distance)
        result = residuals/points
        #result = math.exp(result)-1
        result = angle
        return result


    def featureAverageRows(self, image):
        height, width  = np.shape(image)
        # width = 15, height = 16

        newImage = cv2.resize(image, (width, width), interpolation=cv2.INTER_NEAREST)
        
        #newImage = np.zeros((height, width))
        distances = [0]*width
        
        #newImage = np.zeros((height, width))
        verticalAverages = np.zeros(height)#[0] * height
        for i in range(height-1):
            totalValue = 0
            weightedValue = 0
            for j in range(width):
                totalValue += image[i,j]
                add = image[i,j]*j
                weightedValue += add
            verticalAverages[i] = weightedValue
        verticalAverages *= (width-1)/verticalAverages.max()
        verticalAverages = verticalAverages.astype(int)

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

        matrix = np.matrix(newImage)
        #result = np.cov(newImage,ddof = 0)
        #result = np.abs(result)
        #result = np.average(result)/10000
        result = matrix.conj()
        print()

        #print()
        #result = np.average(downValues)/250*np.average(upValues)/250
        #result = np.dot(downValues, upValues)/600000
        #for column in range(width):
        #    newImage[0, column] = horizontalValues[column]
        
        #newImage[verticalAverages, 5] = 255
        #verticalModel = np.polyfit(range(len(verticalAverages)), verticalAverages, 5)
        
        #draw_y = np.linspace(0, len(verticalAverages)-1, len(verticalAverages))
        #draw_x = np.polyval(verticalModel, draw_y)
        #draw_y = draw_y / 16

 
        #draw_x = np.linspace(0, len(horizontalAverages) -1, len(horizontalAverages))
        #draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)   # needs to be int32 and transposed
        #cv2.polylines(newImage, [draw_points], False, (255), thickness = 1)  # args: image, points, closed, color
        
        
        
        # result = varianceVer/5000
        #result = varianceVer/varianceHor#int(np.dot(verticalAverages, horizontalAverages))/200000-0.5



        # plt.subplot(121),plt.imshow(image, cmap = 'gray')
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122),plt.imshow(newImage, cmap = 'gray')
        # plt.title('New image'), plt.xticks([]), plt.yticks([])
        # plt.show()
        # cv2.waitKey(0)
        return result

    # Features to keep
    # self.featureHorizontalSymmetry(image.copy(), xParameter = 3)
    # self.featureHorizontalSymmetry(image.copy(), xParameter = 8)
    # self.featureIslands(image.copy()) # No parameter required
    # self.featureLaplacian(image.copy()) # No parameter required
    # self.featureFourier(image.copy()) # looks random, but may be usefull
    # self.featureVerticalPolyRow(image.copy())
    # self.featureDiagonalUp(image.copy())

    def featureHorizontalSymmetry(self, image, xParameter):
        image[image >= 20] = 255
        image[image < 20] = 0
        width, height = np.shape(image)
        pixelsLeft = 0
        pixelsRight = 0
        for i in range(width):
            for j in range(height):
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
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        magnitude_spectrum = np.clip(magnitude_spectrum,0,1000)
        result = float(np.average(magnitude_spectrum))/255
        result -= 0.5
        result *= 10
        result += 0.4
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

if __name__ == "__main__":
    program = featurePipeline()
    program.pipeline()