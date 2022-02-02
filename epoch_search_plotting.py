import numpy as np
import matplotlib.pyplot as plt
import csv


num_epochs = []
errors = []
losses = []
with open("epoch_search_100_more_layers.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    
    for line_count, row in enumerate(csv_reader):
        if line_count == 0:
            for item in row:
                num_epochs.append(item)
        elif line_count == 1:
            for item in row:
                errors.append(float(item))
        else:
            for item in row:
                losses.append(float(item))


plt.plot(num_epochs, losses)
plt.show()