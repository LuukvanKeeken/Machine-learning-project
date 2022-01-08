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

        training_data = np.asarray(training_data)
        test_data = np.asarray(test_data)

        # Create arrays containing the labels that correspond to
        # the vectors in the training and testing sets.
        training_labels = []

        for i in range(10):
            training_labels += [i for j in range(100)]

        training_labels = np.asarray(training_labels)
        test_labels = training_labels

        self.dataset = {
            "training_data" : training_data,
            "training_labels" : training_labels,
            "test_data" : test_data,
            "test_labels": test_labels
        }

    # Returns the standard digits dataset
    def digits_standard(self):
        return self.dataset

    # Returns the digits dataset with n_rot extra rotated copies in the range of degrees rot_range
    def digits_rot(self, rot_range, n_rot):
        return self.rand_rotate(self.dataset, rot_range, n_rot)

    # Creates randomly rotated copies of training data
    # rot_range specifies the range of rotation in degrees, e.g. (-10, 10)
    # n_rot specifies the number of extra rotated copies to be added
    def rand_rotate(self, dataset, rot_range, n_rot):
        # Allocates new arrays for data and labels based on n_rot amount of extra rotated copies
        new_data = np.zeros((len(dataset['training_data']) + len(dataset['training_data']) * n_rot, len(dataset['training_data'][0])), dtype=int)
        new_labels = np.zeros(len(dataset['training_data']) + len(dataset['training_labels']) * n_rot, dtype=int)

        # Fills new arrays with randomly rotated images
        counter = 0
        for i in range(len(dataset['training_data'])):
            # Keeps the original sample before adding rotated versions
            new_data[counter] = dataset['training_data'][i]
            new_labels[counter] = dataset['training_labels'][i]
            counter += 1

            # Creates n_rot amount of rotated copies
            for j in range(n_rot):
                image = np.reshape(dataset['training_data'][i], (16,15))
                image_pil = Image.fromarray(image).rotate(np.random.randint(rot_range[0], rot_range[1]))
                new_data[counter] = np.ndarray.flatten(np.array(image_pil))
                new_labels[counter] = dataset['training_labels'][i]
                counter += 1

        return {
            "training_data" : new_data,
            "training_labels" : new_labels,
            "test_data" : dataset['test_data'],
            "test_labels" : dataset['test_labels']
            }
