import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from PIL import Image
from create_data import *
from sklearn.model_selection import KFold
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

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



model = keras.models.load_model('models/best_performing_model_under_1.4')


model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# history = model.fit(training_data, training_labels, epochs = 60)
# scores = model.evaluate(test_data, test_labels)
predictions = model.predict(test_data)
predictions = predictions.argmax(axis=-1)
for i in range(len(predictions)):
    if test_labels[i] != predictions[i]:
        print(f"{test_labels[i]} misclassified as a {predictions[i]}")
        digit = np.reshape(test_data[i], (16,15))
        plt.title(f"{int(test_labels[i])} misclassified as a {predictions[i]}")
        ax = sns.heatmap(digit, cmap='gray_r', cbar=False)
        plt.show()


cm = confusion_matrix(test_labels, predictions)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("CNN confusion matrix")
plt.show()


# print(f"Error: {(1-scores[1])*100}%")
# print(f"Loss: {scores[0]}")