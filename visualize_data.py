from string import digits
from create_data import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

MEAN = 0
VARIANCE = 1
ROTATION = -10
NOISINESS = 0.1
IMAGE_INDEX = 522

datasets = DataSets()

x_train_rtd, _ = datasets.digits_rot(n_copies=1, rot_range=(-10,10))
x_train_noise, _ = datasets.digits_noise(n_copies=1)
x_train_std, _ = datasets.digits_standard()

print("INFO: Size of Original Dataset: {}".format(len(x_train_std)))
print("INFO: Size of Rotated Dataset: {}".format(len(x_train_rtd)))
print("INFO: Size of Noisy Dataset: {}".format(len(x_train_noise)))

fig, axs = plt.subplots(10,10, subplot_kw=dict(xticks=[],yticks=[]))
fig.subplots_adjust(hspace=0.1, wspace=0.01)
row = 0

for i in range(10):
    for j in range(10):
        sample = x_train_std[row + j]
        img = np.reshape(sample, (16,15))
        axs[i,j].imshow(img, cmap='Greys')
    
    row += 100

plt.suptitle("Examples of digits in dataset")
plt.show()


img_std = np.reshape(x_train_std[IMAGE_INDEX], (16, 15))
img_rtd = Image.fromarray(img_std).rotate(ROTATION)
img_rtd = np.array(img_rtd)
img_noise = img_std + NOISINESS * np.random.normal(MEAN, VARIANCE, img_std.shape)

fig, axs = plt.subplots(1,2)

axs[0].imshow(img_std, cmap='Greys')
axs[0].set_title("Original Image")
axs[1].imshow(img_rtd, cmap='Greys')
axs[1].set_title("Rotated by {} Degrees".format(ROTATION))

plt.show()

fig, axs = plt.subplots(1,2)
axs[0].imshow(img_std, cmap='Greys')
axs[0].set_title("Original Image")
axs[1].imshow(img_noise, cmap='Greys')
axs[1].set_title("Gaussian Noise")

plt.show()
