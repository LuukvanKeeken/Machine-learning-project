import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from PIL import Image
from create_data import *
from sklearn.model_selection import KFold, train_test_split
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

num_folds = 1.0

# kfold = KFold(n_splits=int(num_folds), shuffle=True)
num_epochs = 100

accuracies_train = np.zeros(num_epochs)
losses_train = np.zeros(num_epochs)
accuracies_val = np.zeros(num_epochs)
losses_val = np.zeros(num_epochs)

train_x, test_x, train_y, test_y = train_test_split(training_data, training_labels, test_size=0.2)

# for train, test in kfold.split(training_data, training_labels):
model = keras.Sequential([
            layers.Conv2D(31, (2, 2), activation='relu', input_shape=(16, 15, 1)),
            layers.MaxPool2D((2,2)),
            layers.Conv2D(62, (2, 2), activation='relu'),
            layers.MaxPool2D((2,2)),
            layers.Conv2D(62, (2,2), activation='relu'),
            layers.Flatten(),
            layers.Dense(62, activation='relu'),
            layers.Dense(10),
        ])

model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=num_epochs, validation_data=(test_x, test_y))

accuracies_train += history.history["accuracy"]
losses_train += history.history["loss"]
accuracies_val += history.history["val_accuracy"]
losses_val += history.history["val_loss"]


accuracies_train /= num_folds
accuracies_train = (1 - accuracies_train)*100
losses_train /= num_folds

accuracies_val /= num_folds
accuracies_val = (1 - accuracies_val)*100
losses_val /= num_folds


with open("epoch_search_v2_2.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    writer.writerow(range(1, num_epochs+1))
    writer.writerow(accuracies_train)
    writer.writerow(losses_train)
    writer.writerow(accuracies_val)
    writer.writerow(losses_val)
