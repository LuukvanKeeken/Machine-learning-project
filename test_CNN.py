import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from PIL import Image
import csv

activity_alpha = 0.01
kernel_alpha = 0.0000025

training_data = []
training_labels = []
test_data = []
test_labels = []

with open("data/training_data_augmented_degree10_number2.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')

    for row in csv_reader:
        digit = []
        for value in row:
            digit.append(float(value))

        training_data.append(digit)


with open("data/training_labels_augmented_degree10_number2.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')

    for row in csv_reader:
        training_labels.append(float(row[0]))


with open("data/test_data_original.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')

    for row in csv_reader:
        digit = []
        for value in row:
            digit.append(float(value))

        test_data.append(digit)


with open("data/test_labels_original.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')

    for row in csv_reader:
        test_labels.append(float(row[0]))

print(len(training_data))
print(len(test_data))

training_data = np.reshape(training_data, (len(training_data), 16, 15))
training_labels = np.asarray(training_labels)
test_data = np.reshape(test_data, (len(test_data), 16, 15))
test_labels = np.asarray(test_labels)

errors = []
losses = []
for i in range(100):
    print(f"ROUND: {i+1} --------------------------------")
    model = keras.Sequential([
                layers.Conv2D(31, (2, 2), activation='relu', input_shape=(16, 15, 1), kernel_regularizer = regularizers.l2(kernel_alpha)),
                layers.MaxPool2D((2,2)),
                layers.Conv2D(62, (2, 2), activation='relu', kernel_regularizer = regularizers.l2(kernel_alpha)),
                layers.MaxPool2D((2,2)),
                layers.Conv2D(62, (2,2), activation='relu', kernel_regularizer = regularizers.l2(kernel_alpha)),
                layers.Flatten(),
                layers.Dense(62, activation='relu', activity_regularizer = regularizers.l2(activity_alpha), kernel_regularizer = regularizers.l2(kernel_alpha)),
                layers.Dense(10),
            ])

    # model = keras.Sequential([
    #             layers.Conv2D(31, (2, 2), activation='relu', input_shape=(16, 15, 1)),
    #             layers.MaxPool2D((2,2)),
    #             layers.Conv2D(62, (2, 2), activation='relu'),
    #             layers.MaxPool2D((2,2)),
    #             layers.Conv2D(62, (2,2), activation='relu'),
    #             layers.Flatten(),
    #             layers.Dense(62, activation='relu'),
    #             layers.Dense(10),
    #         ])


    model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    history = model.fit(training_data, training_labels, epochs=30, validation_data = (test_data, test_labels))

    scores = model.evaluate(test_data, test_labels)
    errors.append(scores[1])
    losses.append(scores[0])


print(f"Average error rate: {(1-np.mean(errors))*100}%, std dev: {np.std(errors)*100}")
print(f"Average loss: {np.mean(losses)}, std dev: {np.std(losses)}")