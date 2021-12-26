import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Plot first digit for each class in training data.
for i in range(10):   
    test = np.reshape(training_data[100*i], (16,15))
    ax = sns.heatmap(test, cmap='gray_r')
    plt.show()


