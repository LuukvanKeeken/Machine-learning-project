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

num_folds = 5

kfold = KFold(n_splits=num_folds, shuffle=True)


kernel_regs = [0.0000075, 0.00001, 0.000025, 0.00005, 0.000075, 0.0001]
act_regs = [0.0025, 0.005, 0.006, 0.0075, 0.01, 0.025, 0.05]
# kernel_regs = [0, 0.1]
# act_regs = [0, 0.1]

avg_error_rates = []
std_dev_error_rates = []
avg_loss = []
std_dev_loss = []
alpha_combinations = []
counter = 0
for kernel_alpha in kernel_regs:
    for act_alpha in act_regs:
        print(f"CURRENT kernel alpha: {kernel_alpha}, activation alpha: {act_alpha} ")
        counter += 1
        print(f"Combination number {counter}. {(len(kernel_regs)*len(act_regs)) - counter} to go.")
        error_rates = []
        losses = []

        fold_num = 1
        for train, test in kfold.split(training_data, training_labels):
            print(f"FOLD: {fold_num}")

            model = keras.Sequential([
                layers.Conv2D(31, (2, 2), activation='relu', input_shape=(16, 15, 1), kernel_regularizer = regularizers.l2(kernel_alpha)),
                layers.MaxPool2D((2,2)),
                layers.Conv2D(62, (2, 2), activation='relu', kernel_regularizer = regularizers.l2(kernel_alpha)),
                layers.MaxPool2D((2,2)),
                layers.Conv2D(62, (2,2), activation='relu', kernel_regularizer = regularizers.l2(kernel_alpha)),
                layers.Flatten(),
                layers.Dense(62, activation='relu', kernel_regularizer = regularizers.l2(kernel_alpha)),
                layers.Dense(10, activity_regularizer = regularizers.l1(act_alpha)),
            ])

            model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

            history = model.fit(training_data[train], training_labels[train], epochs=60)

            scores = model.evaluate(training_data[test], [training_labels[test]])
            error_rates.append((1-scores[1])*100)
            losses.append(scores[0])

            fold_num += 1

        fold_num = 1
        avg_error_rates.append(np.mean(error_rates))
        std_dev_error_rates.append(np.std(error_rates))
        avg_loss.append(np.mean(losses))
        std_dev_loss.append(np.std(losses))
        alpha_combinations.append(str(kernel_alpha) + ", " + str(act_alpha) )



for i, alpha in enumerate(alpha_combinations):
    print(f"{alpha}:")
    print(f"    Average error rate: {avg_error_rates[i]}")
    print(f"    Average loss: {avg_loss[i]}")

with open("weight_and_activity_alphas_more_precise.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    writer.writerow(alpha_combinations)
    writer.writerow(avg_error_rates)
    writer.writerow(avg_loss)