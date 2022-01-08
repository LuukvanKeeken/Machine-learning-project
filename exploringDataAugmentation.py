import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from PIL import Image
from create_data import *

datasets = DataSets()
dat_std = datasets.digits_standard()
dat_rtd = datasets.rand_rotate(dat_std, (-10,10), 2)

sets = [dat_std, dat_rtd]
alpha = 0.5
counter = 1

for dataset in sets:
    training_data = np.reshape(dataset['training_data'], (len(dataset['training_data']), 16, 15))
    training_labels = dataset['training_labels']
    test_data = np.reshape(dataset['test_data'], (len(dataset['test_data']), 16, 15))
    test_labels = dataset['test_labels']

    model = keras.Sequential([
        layers.Conv2D(31, (2, 2), activation='relu', input_shape=(16, 15, 1)),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(62, (2, 2), activation='relu'),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(62, (2,2), activation='relu'),
        layers.Flatten(),
        layers.Dense(62, activation='relu', activity_regularizer = regularizers.l2(alpha)),
        layers.Dense(10),
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(training_data, training_labels, epochs=30, 
                    validation_data=(test_data, test_labels))

    plt.plot(history.history['val_accuracy'], label = 'val_accuracy' + str(counter))

    counter += 1

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

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