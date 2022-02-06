import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from PIL import Image
# from ..create_data import *
from sklearn.model_selection import KFold, train_test_split
import csv

training_data = []
training_labels = []
test_data = []
test_labels = []

with open("./data/training_data_augmented_degree10_number2.csv") as csv_file:
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


num_folds = 5
num_epochs = 60
min_num_filters = 82
max_num_filters = 120
step_size = 2
kfold = KFold(n_splits=int(num_folds), shuffle=True)


average_errors = []
average_val_errors = []
average_losses = []
average_val_losses = []

for number_of_filters in range(min_num_filters, max_num_filters+1, step_size):
    print(f"NUMBER OF FILTERS: {number_of_filters}")
    errors = []
    val_errors = []
    losses = []
    val_losses = []

    fold_num = 1
    for train, test in kfold.split(training_data, training_labels):
        print(f"NUMBER OF FILTERS: {number_of_filters}, FOLD {fold_num}")

        model = keras.Sequential([
            layers.Conv2D(number_of_filters/2, (2, 2), activation='relu', input_shape=(16, 15, 1)),
            layers.MaxPool2D((2,2)),
            layers.Conv2D(number_of_filters, (2, 2), activation='relu'),
            layers.MaxPool2D((2,2)),
            layers.Conv2D(number_of_filters, (2,2), activation='relu'),
            layers.Flatten(),
            layers.Dense(number_of_filters, activation='relu'),
            layers.Dense(10),
        ])    

        model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        history = model.fit(training_data[train], training_labels[train], epochs=num_epochs)

        accuracy = history.history["accuracy"][-1]
        error = (1 - accuracy)*100
        loss = history.history["loss"][-1]
        errors.append(error)
        losses.append(loss)

        scores = model.evaluate(training_data[test], training_labels[test])
        val_errors.append((1 - scores[1])*100)
        val_losses.append(scores[0])

        fold_num += 1

    average_errors.append(np.mean(errors))
    average_losses.append(np.mean(losses))
    average_val_errors.append(np.mean(val_errors))
    average_val_losses.append(np.mean(val_losses))

with open("filter_search_82-120_lower_first.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    writer.writerow(range(min_num_filters, max_num_filters+1, step_size))
    writer.writerow(average_errors)
    writer.writerow(average_losses)
    writer.writerow(average_val_errors)
    writer.writerow(average_val_losses)


