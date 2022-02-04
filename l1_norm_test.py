import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from PIL import Image
from create_data import *
from sklearn.model_selection import KFold
import csv

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


with open("data/test_data_augmented_degree10_number2.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')

    for row in csv_reader:
        digit = []
        for value in row:
            digit.append(float(value))

        test_data.append(digit)


with open("data/test_labels_augmented_degree10_number2.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')

    for row in csv_reader:
        test_labels.append(float(row[0]))


dataset = {"training_data": np.asarray(training_data),
           "training_labels": np.asarray(training_labels),
           "test_data": np.asarray(test_data),
           "test_labels": np.asarray(test_labels)}


training_data = np.reshape(dataset['training_data'], (len(dataset['training_data']), 16, 15))
training_labels = dataset['training_labels']
test_data = np.reshape(dataset['test_data'], (len(dataset['test_data']), 16, 15))
test_labels = dataset['test_labels']

num_folds = 40

kfold = KFold(n_splits=num_folds, shuffle=True)

fold_num = 1
error_rates = []
losses = []
kernel_reg = 0.0000025
act_reg = 0.01
for train, test in kfold.split(training_data, training_labels):
    print(f"fold: {fold_num}")

    model = keras.Sequential([
        layers.Conv2D(31, (2, 2), activation='relu', input_shape=(16, 15, 1), kernel_regularizer = regularizers.l2(kernel_reg)),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(62, (2, 2), activation='relu', kernel_regularizer = regularizers.l2(kernel_reg)),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(62, (2,2), activation='relu', kernel_regularizer = regularizers.l2(kernel_reg)),
        layers.Flatten(),
        layers.Dense(62, activation='relu', kernel_regularizer = regularizers.l2(kernel_reg)),
        layers.Dense(10, activity_regularizer = regularizers.l1(act_reg)),
    ])

    model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    history = model.fit(training_data[train], training_labels[train], epochs=30)

    scores = model.evaluate(training_data[test], [training_labels[test]])
    error_rates.append((1-scores[1])*100)
    losses.append(scores[0])

    fold_num += 1

# kernel_reg = 0.0000025
# act_reg = 0.01

# model = keras.Sequential([
#         layers.Conv2D(31, (2, 2), activation='relu', input_shape=(16, 15, 1), kernel_regularizer = regularizers.l2(kernel_reg)),
#         layers.MaxPool2D((2,2)),
#         layers.Conv2D(62, (2, 2), activation='relu', kernel_regularizer = regularizers.l2(kernel_reg)),
#         layers.MaxPool2D((2,2)),
#         layers.Conv2D(62, (2,2), activation='relu', kernel_regularizer = regularizers.l2(kernel_reg)),
#         layers.Flatten(),
#         layers.Dense(62, activation='relu', kernel_regularizer = regularizers.l2(kernel_reg), activity_regularizer = regularizers.l1(act_reg)),
#         layers.Dense(10),
#     ])

# model.compile(optimizer='adam',
#             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#             metrics=['accuracy'])

# history = model.fit(training_data, training_labels, epochs = 30)
# scores = model.evaluate(test_data, test_labels)
# print(f"Error: {(1-scores[1])*100}%")
# print(f"Loss: {scores[0]}")


print(f"Average error rate: {np.mean(error_rates)}, std dev: {np.std(error_rates)}")
print(f"Average loss: {np.mean(losses)}, std dev: {np.std(losses)}")
