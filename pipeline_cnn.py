from calendar import EPOCH
from distutils.command.build import build
from tensorflow.keras import Sequential, layers, regularizers, losses
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, KFold
from create_data import *

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

ALPHA = 2.5e-6
EPOCHS = 25

# Returns model with specified parameters.
# Note: n_filters determines the number of filters in the input layer
def build_model(n_filters=32, n_layers=3, act='relu', f_size=(2,2), reg=None, pad='valid'):
    model = Sequential()

    # Input layer
    model.add(layers.Conv2D(n_filters, f_size, activation=act, input_shape=(16, 15, 1)))
    model.add(layers.MaxPooling2D(f_size))

    # Determine number of additional layers to add
    n_layers -= 1
    r = range(1, n_layers + 1)

    # Add layers with appropriate number of filters
    for i in r:
        #model.add(layers.Conv2D(n_filters * 2 ** i, f_size, activation=act, padding=pad))
        model.add(layers.Conv2D(n_filters, f_size, activation=act, padding=pad))
        model.add(layers.MaxPooling2D(f_size, padding=pad))

    # Add output layers
    model.add(layers.Flatten())
    #model.add(layers.Dense(n_filters * 2 ** i, activation=act, activity_regularizer=reg))
    model.add(layers.Dense(n_filters, activation=act, activity_regularizer=reg))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return model

# Performs k fold cross-validation with k folds with e epochs for each fold
def cross_validation(model, x_val, y_val, k=3, e=25):
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    fold = 1
    accuracies = []

    for train_index, test_index in kf.split(x_val):
        x_train, x_test = x_val[train_index], x_val[test_index]
        y_train, y_test = y_val[train_index], y_val[test_index]

        model.fit(x_train, y_train, epochs=e, verbose=0)
        accuracy = evaluate(model, x_test, y_test, acc=True, f1=False, cm=False)
        accuracies.append(accuracy)

        print("INFO: Fold {} completed with accuracy: {}".format(fold, accuracy))

        fold += 1

    average = np.average(accuracies)

    print("INFO: {} fold cross-validation completed with average accuracy: {}".format(k, average))

    return average

# Evaluates specified model with accuracy (acc=true), f1 scores (f1=true), and confusion matrix (cm=true)
# Requires a pre-trained model
def evaluate(model, testing_data, testing_labels, acc=True, f1=True, cm=True, print=False):
    labels = np.unique(testing_labels)

    predictions = model.predict(testing_data)
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(testing_labels, predictions)
    f1_scores = f1_score(testing_labels, predictions, average=None)
    c_matrix = confusion_matrix(testing_labels, predictions)

    output = []

    if acc:
        if print:
            print("INFO: Total accuracy is: {}\n".format(accuracy))
        output.append(accuracy)
    if f1:
        if print:
            print("INFO: F1 scores: {}".format(f1_scores))
        output.append(f1_scores)
    if cm:
        c_matrix = c_matrix / c_matrix.astype(float).sum(axis=1)
        df_cm = pd.DataFrame(c_matrix, index = labels, columns = labels)

        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, cmap='viridis')
        plt.title("Confusion Matrix")
        plt.ylabel("True Labels")
        plt.xlabel("Predicted Labels")
        plt.show()

    if len(output) == 1:
        return output[0]
    else:
        return output

# Tests CNN model with number of filters specified in filter_range
def test_layer_size(filter_range, x_train, y_train, x_test, y_test):
    accuracy = []
    n_avg = 1

    print("INFO: Testing different number of filters.\n")

    for n in filter_range:
        avg = []

        for i in range(n_avg):
            print("INFO: Testing model with {} filters.".format(n))

            model = build_model(n_filters=n)
            model.fit(x_train, y_train, epochs=EPOCHS, verbose=0)
            avg.append(evaluate(model, x_test, y_test, acc=True, cm=False, f1=False))

        accuracy.append(np.average(avg))

    plt.plot(filter_range, accuracy)
    plt.title("Number of Output Filters vs. Model Accuracy")
    plt.xlabel("Number of Filters")
    plt.ylabel("Accuracy")
    plt.ylim([0,1])
    plt.show()

# Plots the training and testing loss of model over e epochs
def test_overfitting(model, x_train, y_train, x_test, y_test, e=25, v=0):
    print("INFO: Analyzing overfitting.")

    history = model.fit(x_train, y_train, epochs=e, 
                            validation_data=(x_test, y_test), 
                            verbose=v)

    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title("Model Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(['Train', 'Test'])
    plt.show()

def main():
    datasets = DataSets()
    #x_train, y_train, x_val, y_val = datasets.digits_noise(n_copies=4)
    #x_train, y_train = datasets.digits_rot(n_copies=1, rot_range=(-10,10))
    x_train, y_train = datasets.digits_standard()

    #x_test, y_test = datasets.digits_testing()

    # Reshape to work with tf models
    x_train = np.reshape(x_train, (len(x_train), 16, 15))
    #x_test = np.reshape(x_test, (len(x_test), 16, 15))

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.7, random_state=1)

    f_range = list(range(1, 10 + 1, 1))
    # Zero padding is introduced for testing layer sizes > 3
    # n_filters is the amount of filters in the conv layers AS WELL as the amount of nodes in dense layer
    model = build_model(n_filters=8, n_layers=3, pad='valid')

    test_overfitting(model, x_train, y_train, x_test, y_test, e=400, v=1)
    #test_layer_size(f_range, x_train, y_train, x_test, y_test)
    #cross_validation(model, x_train, y_train, k=10, e=10)

    model.summary()

main()

