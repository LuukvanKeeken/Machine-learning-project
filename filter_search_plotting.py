import numpy as np
import matplotlib.pyplot as plt
import csv


num_epochs = []
errors_train = []
losses_train = []
errors_val = []
losses_val = []
# with open("filter_search_2-80_lower_first.csv") as csv_file:
with open("filter_search_2-120_lower_first.csv") as csv_file:
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



# plt.plot(num_epochs[3:-1], errors_train[3:-1], label= "training error")
# plt.plot(num_epochs[3:-1], errors_val[3:-1], label= "validation error")
# plt.title("5-fold CV filter number search")
# plt.legend()
# plt.xlabel("Number of filters")
# plt.ylabel("Average error")
# plt.show()

plt.plot(num_epochs[3:-1], losses_train[3:-1], label= "training loss")
plt.plot(num_epochs[3:-1], losses_val[3:-1], label= "validation loss")
# plt.xticks(range(2, 7, 2))
plt.title("5-fold CV filter number search first lower")
plt.legend()
plt.xlabel("Number of filters")
plt.ylabel("Average loss")
plt.show()