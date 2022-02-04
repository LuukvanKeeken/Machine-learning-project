import numpy as np
import matplotlib.pyplot as plt
import csv


num_epochs = []
errors_train = []
losses_train = []
errors_val = []
losses_val = []
with open("epoch_search_1_run2.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    
    for line_count, row in enumerate(csv_reader):
        if line_count == 0:
            for item in row:
                num_epochs.append(item)
        elif line_count == 1:
            for item in row:
                errors_train.append(float(item))
        elif line_count == 2:
            for item in row:
                losses_train.append(float(item))
        elif line_count == 3:
            for item in row:
                errors_val.append(float(item))
        else:
            for item in row:
                losses_val.append(float(item))



plt.plot(num_epochs[3:-1], errors_train[3:-1], label= "training loss")
plt.plot(num_epochs[3:-1], errors_val[3:-1], label= "validation loss")
plt.xticks(range(1, 101, 5))
plt.title("One run with 100 epochs")
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("Average loss")
plt.show()