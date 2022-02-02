import csv
from create_data import *

datasets = DataSets()
dat_std = datasets.digits_standard()
dat_rtd = datasets.rand_rotate(dat_std, (-10,10), 2)

with open("data/training_data_original.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    for i in range(len(dat_std["training_data"])):
        writer.writerow(dat_std["training_data"][i])


with open("data/training_labels_original.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    for i in range(len(dat_std["training_labels"])):
        writer.writerow([dat_std["training_labels"][i]])


with open("data/test_data_original.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    for i in range(len(dat_std["test_data"])):
        writer.writerow(dat_std["test_data"][i])


with open("data/test_labels_original.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    for i in range(len(dat_std["test_labels"])):
        writer.writerow([dat_std["test_labels"][i]])


with open("data/training_data_augmented_degree10_number2.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    for i in range(len(dat_rtd["training_data"])):
        writer.writerow(dat_rtd["training_data"][i])


with open("data/training_labels_augmented_degree10_number2.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    for i in range(len(dat_rtd["training_labels"])):
        writer.writerow([dat_rtd["training_labels"][i]])


with open("data/test_data_augmented_degree10_number2.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    for i in range(len(dat_rtd["test_data"])):
        writer.writerow(dat_rtd["test_data"][i])


with open("data/test_labels_augmented_degree10_number2.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    for i in range(len(dat_rtd["test_labels"])):
        writer.writerow([dat_rtd["test_labels"][i]])