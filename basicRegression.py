import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.loadtxt("mfeat-pix.txt", dtype='i')  # 2000 rows, 240 columns

    trainIndices = [*range(0, 100), *range(200, 300), *range(400, 500), *range(600, 700), *range(800, 900),
                    *range(1000, 1100), *range(1200, 1300), *range(1400, 1500), *range(1600, 1700), *range(1800, 1900)]
    testIndices = [index + 100 for index in trainIndices]
    trainPatterns = data[trainIndices]  # 1000 rows, 240 columns
    testPatterns = data[testIndices]

    b = np.ones(100, dtype=int)
    correctLabels = np.concatenate((np.zeros(100, dtype=int), b, b*2, b*3, b*4, b*5, b*6, b*7, b*8, b*9))

    meanTrainImages = np.zeros((240, 10))
    for i in range(10):
        for j in range(240):
            meanTrainImages[j, i] = np.mean(trainPatterns[i*100:(i+1)*100, j])

    print(meanTrainImages)
    fig, axs = plt.subplots(nrows=2, ncols=5)
    for i in range(10):
        image = np.zeros((16, 15))
        index = 0
        for row in range(16):
            for col in range(15):
                image[row, col] = meanTrainImages[index, i]
                index += 1

        if i < 5:
            axs[0, i].imshow(image, cmap='gray', vmin=0, vmax=6)
        else:
            axs[1, i-5].imshow(image, cmap='gray', vmin=0, vmax=6)

    plt.show()
