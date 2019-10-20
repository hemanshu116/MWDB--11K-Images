import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from Main import config


def progress(count, total):
    percent = (count / total) * 100
    print('\r', "%.2f" % round(percent, 2) + "% completed", end=' ')


# For task 2 and 4
def printMatch(finalList, k):
    sortList = sorted(finalList.items(), key=lambda x: x[1])
    sortList = dict(sortList)
    forNormalize = []

    for image, score in sortList.items():
        forNormalize.append(score)
    afterNormal = list(normalize_score(forNormalize))
    i = 0
    for image in sortList:
        if i > k:
            break
        sortList[image] = (1 - afterNormal[i]) * 100
        print(image + " : " + str(sortList[image]) + " % match")
        i = i + 1

    plot(k, sortList)


def findDistance(selectedImage, latentFeatureDict):
    output = {}
    for key, value in latentFeatureDict.items():
        imageFromDatabase = np.asarray(latentFeatureDict[key])
        output[key] = np.linalg.norm(selectedImage - imageFromDatabase)
    return output


def normalize_score(data):
    scaler = MinMaxScaler()
    data_array = np.asarray(data).reshape((len(data), 1))
    scaled_values = scaler.fit_transform(data_array)
    return pd.Series(scaled_values.reshape(len(scaled_values)))


def plot(num_images, imageScores):
    # loop over the results
    imageScores = list(imageScores.items())
    fig = plt.figure(figsize=(20, 20))
    columns = 3
    rows = int(round(num_images / 3) + 1)
    for i in range(1, columns * rows + 1):
        if i > num_images:
            break
        # print(imageScores[i-1])
        file_name, distance = imageScores[i - 1]
        img = cv2.imread(str(config.IMAGE_FOLDER+ "\\" + file_name))
        ax = fig.add_subplot(rows, columns, i)
        ax.set_title(file_name + "_" + str("{0:.2f}".format(distance)))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
