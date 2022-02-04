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

# num_folds = 5

# kfold = KFold(n_splits=num_folds, shuffle=True)

# avg_error_rates = []
# std_dev_error_rates = []
# avg_loss = []
# std_dev_loss = []
# min_num_epochs = 61
# max_num_epochs = 100

# for num_epochs in range(min_num_epochs, max_num_epochs + 1, 2):
    
#     print(f"CURRENT NUMBER OF EPOCHS: {num_epochs} -------------------------------------------")
#     error_rates = []
#     losses = []


#     fold_num = 1
#     for train, test in kfold.split(training_data, training_labels):
#         print(f"# of epochs: {num_epochs}, fold: {fold_num}")

#         model = keras.Sequential([
#             layers.Conv2D(31, (2, 2), activation='relu', input_shape=(16, 15, 1)),
#             layers.MaxPool2D((2,2)),
#             layers.Conv2D(62, (2, 2), activation='relu'),
#             layers.MaxPool2D((2,2)),
#             layers.Conv2D(62, (2,2), activation='relu'),
#             layers.Flatten(),
#             layers.Dense(62, activation='relu'),
#             layers.Dense(10),
#         ])

#         model.compile(optimizer='adam',
#                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])

#         history = model.fit(training_data[train], training_labels[train], epochs=num_epochs)

#         scores = model.evaluate(training_data[test], [training_labels[test]])
#         error_rates.append((1-scores[1])*100)
#         losses.append(scores[0])

#         fold_num += 1
    
#     fold_num = 1
#     avg_error_rates.append(np.mean(error_rates))
#     std_dev_error_rates.append(np.std(error_rates))
#     avg_loss.append(np.mean(losses))
#     std_dev_loss.append(np.std(losses))

# with open("epoch_search_100.csv", mode="w", newline='') as csv_file:
#     writer = csv.writer(csv_file, delimiter=",")

#     writer.writerow(range(min_num_epochs, max_num_epochs + 1, 2))
#     writer.writerow(avg_error_rates)
#     writer.writerow(avg_loss)

train_x, test_x, train_y, test_y = train_test_split(training_data, training_labels, test_size = 0.2)

model = keras.Sequential([
            layers.Conv2D(64, (2, 2), activation='relu', input_shape=(16, 15, 1)),
            layers.MaxPool2D((2,2)),
            layers.Conv2D(64, (2, 2), activation='relu'),
            # layers.MaxPool2D((2,2)),
            # layers.Conv2D(64, (2,2), activation='relu'),
            # layers.Conv2D(64, (2,2), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            # layers.Dense(16, activation='relu'),
            layers.Dense(10),
        ])


model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
# print(model.summary())
# exit()
history = model.fit(train_x, train_y, epochs=100, validation_data = (test_x, test_y))

error_rate = (1-np.asarray(history.history["accuracy"]))*100
error_rate_val = (1-np.asarray(history.history["val_accuracy"]))*100

with open("epoch_search_1_run2.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    writer.writerow(range(1, 101))
    writer.writerow(error_rate)
    writer.writerow(history.history["loss"])
    writer.writerow(error_rate_val)
    writer.writerow(history.history["val_loss"])