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


num_folds = 2.0
num_epochs = 200


kfold = KFold(n_splits=int(num_folds), shuffle=True)

fold_num = 1
error_rate = np.zeros(num_epochs)
error_rate_val = np.zeros(num_epochs)
loss = np.zeros(num_epochs)
loss_val = np.zeros(num_epochs)

for train, test in kfold.split(training_data, training_labels):
    print(f"FOLD {fold_num}")

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

    history = model.fit(training_data[train], training_labels[train], epochs=num_epochs, validation_data = (training_data[test], training_labels[test]))

    new_error_rate = (1 - np.asarray(history.history["accuracy"]))*100
    new_error_rate_val = (1 - np.asarray(history.history["val_accuracy"]))*100

    error_rate += new_error_rate
    error_rate_val += new_error_rate_val
    loss += history.history["loss"]
    loss_val += history.history["val_loss"]

    fold_num += 1

error_rate /= num_folds
error_rate_val /= num_folds
loss /= num_folds
loss_val /= num_folds

with open("new_epoch_search_2fold2.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    writer.writerow(range(1, num_epochs+1))
    writer.writerow(error_rate)
    writer.writerow(history.history["loss"])
    writer.writerow(error_rate_val)
    writer.writerow(history.history["val_loss"])
    