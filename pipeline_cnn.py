from tensorflow.keras import Sequential, layers, regularizers, losses
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from create_data import *

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

datasets = DataSets()

x_train, y_train, x_test, y_test = datasets.digits_rot(n_copies=2, rot_range=(-10,10))

# Reshape to work with tf models
x_train = np.reshape(x_train, (len(x_train), 16, 15))
x_test = np.reshape(x_test, (len(x_test), 16, 15))

ALPHA = 2.5e-6
EPOCHS = 25

# Evaluates specified model with accuracy (acc=true), f1 scores (f1=true), and confusion matrix (cm=true)
def evaluate(model, testing_data, testing_labels, acc=True, f1=True, cm=True):
    labels = np.unique(testing_labels)

    predictions = model.predict(testing_data)
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(testing_labels, predictions)
    f1_scores = f1_score(testing_labels, predictions, average=None)
    c_matrix = confusion_matrix(testing_labels, predictions)

    output = []

    if(acc):
        print("INFO: Total accuracy is: {}\n".format(accuracy))
        output.append(accuracy)
    if(f1):
        print("INFO: F1 scores: {}".format(f1_scores))
        output.append(f1_scores)
    if(cm):
        c_matrix = c_matrix / c_matrix.astype(float).sum(axis=1)
        df_cm = pd.DataFrame(c_matrix, index = labels, columns = labels)

        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, cmap='viridis')
        plt.title("Confusion Matrix")
        plt.ylabel("True Labels")
        plt.xlabel("Predicted Labels")
        plt.show()

    return output

# Tests CNN model with number of filters specified in filter_range
def test_layer_size(filter_range):
    accuracy = []
    
    print("INFO: Testing different number of filters.\n")
    
    n_avg = 5

    for n in filter_range:
        avg = []

        for i in range(n_avg):
            print("INFO: Testing model with {} filters.".format(n))
            model = Sequential()
            model.add(layers.Conv2D(n / 2, (2, 2), activation='relu', input_shape=(16, 15, 1)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(n, (2, 2), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(n * 2, (2, 2), activation='relu'))
            model.add(layers.Flatten())
            model.add(layers.Dense(n, activation='relu', activity_regularizer = regularizers.l2(ALPHA)))
            model.add(layers.Dense(10))

            model.compile(optimizer='adam',
                            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

            model.fit(x_train, y_train, epochs=EPOCHS, verbose=0)

            avg.append(evaluate(model, x_test, y_test, acc=True, cm=False, f1=False))

        accuracy.append(np.average(avg))

    plt.plot(filter_range, accuracy)
    plt.title("Number of Output Filters vs. Model Accuracy")
    plt.xlabel("Number of Filters")
    plt.ylabel("Accuracy")
    plt.ylim([0,1])
    plt.show()

def main():
    f_range = list(range(2, 22 + 2, 2))
    test_layer_size(f_range)

main()

