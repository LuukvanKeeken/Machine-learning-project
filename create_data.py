import numpy as np
from PIL import Image

class DataSets:
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
        training_data = []
        test_data = []
        counter = 0

        for k in range(10):
            for i in range(2):
                for j in range(100):
                    if i == 0:
                        training_data.append(data[counter])
                    else:
                        test_data.append(data[counter])
                    counter += 1

        self.training_data = np.asarray(training_data)
        self.test_data = np.asarray(test_data)

        # Create arrays containing the labels that correspond to
        # the vectors in the training and testing sets.
        training_labels = []

        for i in range(10):
            training_labels += [i for j in range(100)]

        self.training_labels = np.asarray(training_labels)
        self.test_labels = training_labels

    # Returns the standard digits dataset
    def digits_standard(self):
        return self.training_data, self.training_labels

    # Returns the testing data from the original dataset
    def digits_testing(self):
        return self.test_data, self.test_labels

    # # Returns an augmented version of the digits dataset with n_rot extra rotated copies in the range of degrees rot_range
    def digits_rot(self, n_copies=2, rot_range=(-10,10)):
        extra_data = np.zeros((len(self.training_data) * n_copies, len(self.training_data[0])), dtype=int)
        extra_labels = np.zeros(len(self.training_labels) * n_copies, dtype=int)
        counter = 0

        for i, s in enumerate(self.training_data):
            for n in range(n_copies):
                image = np.reshape(s, (16,15))
                image_pil = Image.fromarray(image).rotate(np.random.randint(rot_range[0], rot_range[1]))
                image = np.ndarray.flatten(np.array(image_pil))
                extra_data[counter] = image
                extra_labels[counter] = self.training_labels[i]
                counter += 1
        
        return extra_data, extra_labels

    # Returns an augmented version of the digits dataset with n_copies extra copies with random Gaussian noise
    def digits_noise(self, n_copies=2, mean=0, var=1):
        extra_data = np.zeros((len(self.training_data) * n_copies, len(self.training_data[0])), dtype=int)
        extra_labels = np.zeros(len(self.training_labels) * n_copies, dtype=int)
        counter = 0

        for i, s in enumerate(self.training_data):
            for j in range(n_copies):
                noise = np.random.normal(mean, var, s.shape)
                extra_data[counter] = s + noise
                extra_labels[counter] = self.training_labels[i]
                counter += 1

        return extra_data, extra_labels
