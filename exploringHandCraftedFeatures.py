import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
import scipy.stats as stats

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
        featureResult = np.zeros((10,100))

        # Plot first digit for each class in training data.
        for i in range(10):
            for j in range(100):
                index = 100*i + j
                image = np.reshape(self.training_data[index], (16,15))
                image *= int(255/image.max())
                image = image.astype(np.uint8)
                image = (255-image)
                label = self.training_labels[index]
                featureResult[i,j] += self.feature(image)
 
        
            

        self.plotFeatureResult(featureResult)
        print()

    def featureInk(self, image):
        result = np.average(image)/255
        return result

    def featureIslands(self, image):
        result = np.average(image)/255
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
        result = modelTwo[1]/280
        # draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)   # needs to be int32 and transposed
        # cv2.polylines(image, [draw_points], False, (100), thickness = 1)  # args: image, points, closed, color
        # scale = 30
        # image = cv2.resize(image, (15*scale, 16*scale))
        # cv2.imshow("test",image) #Original image
        # cv2.waitKey(0) 
        return result

    def feature(self, image, testlabel = 1):
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
        symmetry = pixelsTop / total
        result = symmetry




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

        return result

    def plotFeatureResult(self,featureResult):
        x = np.linspace(-0.4, 1.5, 100)
        for i in range(10):
            print(np.min(featureResult))
            print(np.max(featureResult))
            mu = np.average(featureResult[i])
            sigma = np.std(featureResult[i])


            #x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), label = str(i))
            

            #value = featureResult[i,0] / featureResult[i,1]
            value = 1
            value *= 100
            line = str(i) + ": "
            for j in range(int(value)):
                line += "."

            #print(line)
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
