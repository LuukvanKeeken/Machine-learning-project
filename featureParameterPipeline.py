from re import I, L
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.mixture import GaussianMixture
import math
from create_data import DataSets
from HCFeatures import HCFeatures
from matplotlib.ticker import MaxNLocator
import string
from scipy.interpolate import interp1d
from sklearn import linear_model

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
            axi.yaxis.set_major_locator(MaxNLocator(integer=True))
            axi.text(-0.1, 1.1, string.ascii_uppercase[i], transform=axi.transAxes, size=20, weight='bold')

            
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
                       
                handles, labels = axi.get_legend_handles_labels()

        plt.legend(handles = handles, labels = labels, loc='upper center', 
             bbox_to_anchor=(-1.5, -0.2),fancybox=False, shadow=False,
             ncol=10)
            
        #for i, axi in enumerate(ax.flat):
        #    axi.set_ylim(0, maxY)
        # for i in [2, 3]:
        #     plt.delaxes(ax.flatten()[i])
        #plt.legend()
        #plt.tick_params(axis='both', which='major', labelsize=5)
        
        plt.savefig("HCfeatures.png", dpi = 300, bbox_inches='tight') # when saving, specify the DPI
        #plt.show()
        
        print() 
    
    def plotFourier(self, image):
        image = np.reshape(image, (16,15))
        image = 6-image
        row = image[13]#11

        fig, ax = plt.subplots(2, 3, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        ax.flat[0].imshow(image,cmap = 'gray')
        ax.flat[1].plot(row)

        x = np.linspace(0, len(row)-1, len(row))#, endpoint=True)
        y = row
        xnew = np.linspace(0, len(row)-1, 100)
        f_linear = interp1d(x, y)
        f_cubic = interp1d(x, y, kind='cubic')

        #sineRow = f_cubic(xnew)
        #ax.flat[1].plot(sineRow)
        ax.flat[1].plot(xnew, f_cubic(xnew), '--', label='cubic')
        dft = cv2.dft(np.float32(row),flags = cv2.DFT_COMPLEX_OUTPUT)
        #dft = np.fft.fft(row)/len(row)
        #ax.flat[2].plot(dft)
        ax.flat[2].plot(dft[:,:,0]) # at 2 and 13 are peeks of (7, 7) and (7,-7)

        #ax.flat[2].plot(dft[:,:,1])
        #magnitude_spectrum = 20*np.log(cv2.magnitude(dft[:,:,0],dft[:,:,1])+1e-15)
        magnitude_spectrum = cv2.magnitude(dft[:,:,0],dft[:,:,1]) # 13 DC and 10, 10
        ax.flat[2].plot(x, [7.1]*15, '--')
        #ax.flat[2].plot(magnitude_spectrum)
        
        base = 2*math.pi/15 * (xnew+1)
        scale = 2.29/(len(row))
        ax.flat[3].plot(xnew, np.cos(base*2)*7*scale)
        ax.flat[3].plot(xnew, np.cos(base*13)*scale*7*2/13)
        ax.flat[3].plot(x, [0.8]*15)

        ax.flat[4].plot(xnew, (np.cos(base*2)*7 + np.cos(base*13)*7*2/13)*scale+1)
        #ax.flat[3].plot(xnew, np.sin(2*xnew))
        
        #result = float(np.average(dft[:,:,0]))
        #result = float(np.average(magnitude_spectrum))



        plt.show()
        print()

    def plotClassificationResult(self, wrongDigits, totalDigits, title):
        fig, ax = plt.subplots()

        #ax = fig.add_axes([0,0,1,1])
        #wrongDigits/=(len(wrongDigits)/totalDigits)
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
        plt.savefig("classificationFeatureRegression.png", dpi = 300, bbox_inches='tight')
        plt.show()
        print()



    def pipeline(self):
        experimentResults = []
 
        createData = DataSets()
        trainingData = createData.digits_standard()
        #self.plotFourier(trainingData[0][640])

        features = HCFeatures()
        trainingRequired = features.trainingRequired
        #features.trainMoG(trainingData)
        for i in range(100):
            print(i)
            features.trainMoG(trainingData)
        features.trainMeanImages(trainingData)
            

        trainingX, _ = trainingData
        
        
        regressionX = np.zeros((1000,18))
        regressionY = np.zeros((1000))
        featuresResult = np.zeros((18, 10,100))
        for index, predictX in enumerate(trainingX):
            digit = int(index/100)
            number = index - digit*100
            featureVector = features.predict(predictX)
            featuresResult[:, digit, number] = featureVector
            regressionX[index] = featureVector
            regressionY[index] = digit
        
        regr = linear_model.LinearRegression()
        regr.fit(regressionX, regressionY)

        good = 0
        wrongDigits = np.zeros(10)
        testingX, testingY = createData.digits_testing()
        for index, predictX in enumerate(testingX):
            featureVector = np.array(features.predict(predictX))
            featureVector = featureVector.reshape(1, -1)
            predictDigit = regr.predict(featureVector)
            realDigit = testingY[index]
            if np.round(predictDigit) == realDigit:
                good +=1
            else:
                wrongDigits[realDigit] +=1
        accuracy = good/len(testingX)
        print("accuracy of linear regression on features is " + str(accuracy))

        self.plotClassificationResult(wrongDigits, 1000, "Error linear regression on HC features")

        experimentResults.append(["Horizontal symmetry x=3",featuresResult[0]])
        experimentResults.append(["Horizontal symmetry x=8",featuresResult[1]])
        experimentResults.append(["Islands",featuresResult[2]])
        experimentResults.append(["Laplacian",featuresResult[3]])
        experimentResults.append(["Fourier",featuresResult[4]])
        experimentResults.append(["Regression on row averages",featuresResult[5]])
        experimentResults.append(["Mixture of Gaussians",featuresResult[6]])
        experimentResults.append(["Average pixels value",featuresResult[7]])

        self.plotExperimentResult(experimentResults)
    

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


if __name__ == "__main__":
    program = featurePipeline()
    program.pipeline()