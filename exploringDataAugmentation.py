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


alphas = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2.5]
num_folds = 20
dataset = {"training_data": np.asarray(training_data),
           "training_labels": np.asarray(training_labels),
           "test_data": np.asarray(test_data),
           "test_labels": np.asarray(test_labels)}


training_data = np.reshape(dataset['training_data'], (len(dataset['training_data']), 16, 15))
training_labels = dataset['training_labels']
test_data = np.reshape(dataset['test_data'], (len(dataset['test_data']), 16, 15))
test_labels = dataset['test_labels']

kfold = KFold(n_splits=num_folds, shuffle=True)

avg_error_rates = []
std_dev_error_rates = []
avg_loss = []
std_dev_loss = []
for alpha in alphas:
    print(f"CURRENT ALPHA: {alpha} -------------------------------------------")
    error_rates = []
    losses = []

    fold_num = 1
    for train, test in kfold.split(training_data, training_labels):
        print(f"alpha: {alpha}, fold: {fold_num}")

        model = keras.Sequential([
            layers.Conv2D(31, (2, 2), activation='relu', input_shape=(16, 15, 1), kernel_regularizer = regularizers.l2(0)),
            layers.MaxPool2D((2,2)),
            layers.Conv2D(62, (2, 2), activation='relu', kernel_regularizer = regularizers.l2(0)),
            layers.MaxPool2D((2,2)),
            layers.Conv2D(62, (2,2), activation='relu', kernel_regularizer = regularizers.l2(0)),
            layers.Flatten(),
            layers.Dense(62, activation='relu', activity_regularizer = regularizers.l2(alpha), kernel_regularizer = regularizers.l2(0.0001)),
            layers.Dense(10),
        ])

        model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

        history = model.fit(training_data[train], training_labels[train], epochs=30)

        scores = model.evaluate(training_data[test], [training_labels[test]])
        error_rates.append((1-scores[1])*100)
        losses.append(scores[0])

        fold_num += 1

    fold_num = 1
    avg_error_rates.append(np.mean(error_rates))
    std_dev_error_rates.append(np.std(error_rates))
    avg_loss.append(np.mean(losses))
    std_dev_loss.append(np.std(losses))

    # plt.plot(history.history['val_accuracy'], label = 'val_accuracy' + str(counter))

for i, alpha in enumerate(alphas):
    print(f"{alpha}:")
    print(f"    Average error rate: {avg_error_rates[i]}")
    print(f"    Average loss: {avg_loss[i]}")

with open("scores_K_20_augmented_best_kernel_comb_last_activity.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    writer.writerow(alphas)
    writer.writerow(avg_error_rates)
    writer.writerow(avg_loss)


# Visualizing difference between rotated digits

# image = np.reshape(dat_std['training_data'][813], (16,15))

# fig, axs = plt.subplots(2,2)
# fig.suptitle("Rotated Images")

# axs[0,0].imshow(image)
# axs[0,0].set_title("Original Image")

# image_pil = Image.fromarray(image)
# image_pil = image_pil.rotate(4)
# image = np.array(image_pil)

# axs[0,1].imshow(image)
# axs[0,1].set_title("Rotated by 10 degrees")

# image_pil = Image.fromarray(image)
# image_pil = image_pil.rotate(5)
# image = np.array(image_pil)

# axs[1,0].imshow(image)
# axs[1,0].set_title("Rotated by 15 degrees")

# image_pil = Image.fromarray(image)
# image_pil = image_pil.rotate(15)
# image = np.array(image_pil)

# axs[1,1].imshow(image)
# axs[1,1].set_title("Rotated by 20 degrees")

#plt.show()