import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers



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
# for i in range(10):   
#     test = np.reshape(training_data[100*i], (16,15))
#     ax = sns.heatmap(test, cmap='gray_r')
#     plt.show()

# Train a very simple CNN on the training data, validate on the test data.
# tf.random.set_seed(15)
training_data, test_data = training_data/6.0, test_data/6.0
training_data = np.reshape(training_data, (1000, 16, 15))
test_data = np.reshape(test_data, (1000, 16, 15))

best_alphas = []
best_values = []
zero_alpha_values = []
for i in range(3):
    print(f"ROUND: {i}")
    print("----------------------------------------------------")
    # Try out various alpha values for the regularization. Plot 
    # the error rates for each of them, and print them.
    alphas = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    smallest = 100
    smallest_alpha = 0
    all_values = []
    for alpha in alphas:
        print(f"Alpha: {alpha}")
        model = models.Sequential()
        model.add(layers.Conv2D(31, (2, 2), activation='relu', input_shape=(16, 15, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(62, (2, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(62, (2, 2), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(62, activation='relu', activity_regularizer = regularizers.l2(alpha)))
        model.add(layers.Dense(10))
        # print(model.summary())

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        history = model.fit(training_data, training_labels, epochs=10, 
                            validation_data=(test_data, test_labels))

        test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)

        print(f"Error rate: {(1-test_acc)*100}%")
        if (1-test_acc)*100 < smallest:
            smallest = (1-test_acc)*100
            smallest_alpha = alpha
        all_values.append((1-test_acc)*100)

    print(f"Smallest error rate is {smallest} for alpha = {smallest_alpha}.")
    alphas_np = np.asarray(alphas)
    all_values_np = np.asarray(all_values)
    inds = all_values_np.argsort()
    sorted_alphas = alphas_np[inds]
    all_values_np.sort()
    print("Sorted alphas:")
    for i in range(len(alphas)):
        print(f"{i}) {sorted_alphas[i]}: {all_values_np[i]}")

    # plt.plot(alphas, all_values)
    # plt.show()
    best_alphas.append(sorted_alphas[0])
    best_values.append(all_values_np[0])
    
    for i in range(len(alphas)):
        if sorted_alphas[i] == 0:
            zero_alpha_values.append(all_values_np[i])

print('The best alpha values for each round:')
print(best_alphas)
best_alphas = np.asarray(best_alphas)
print(f"Mean: {best_alphas.mean()}, stddev: {best_alphas.std()}")
print('The corresponding best error rates')
print(best_values)
best_values = np.asarray(best_values)
print(f"Mean: {best_values.mean()}, stddev: {best_values.std()}")
print('The error rates when using alpha of 0')
print(zero_alpha_values)
zero_alpha_values = np.asarray(zero_alpha_values)
print(f"Mean: {zero_alpha_values.mean()}, stddev: {zero_alpha_values.std()}")
