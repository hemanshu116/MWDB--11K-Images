import json
import os
from os.path import join

import cv2
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
import matplotlib.pyplot as plt

from Main import config


def progress(count, total):
    percent = (count / total) * 100
    print('\r', "%.2f" % round(percent, 2) + "% completed", end=' ')


# For task 2 and 4
def printMatch(finalList, k, outputFolderParams):
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

    plot(k, sortList, outputFolderParams, forTask1and3=False)


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


def plot(num_images, imageScores, outputFolderParams, forTask1and3=True, finalImageName=1):
    # loop over the results
    imageScores = list(imageScores.items())
    if forTask1and3:
        columns = 15
        fig = plt.figure(figsize=(70, 70))
    else:
        columns = 3
        fig = plt.figure(figsize=(20, 20))
    rows = int(round(num_images / 3) + 1)
    for i in range(1, columns * rows + 1):
        if i > num_images:
            break
        # print(imageScores[i-1])
        filename, distance = imageScores[i - 1]
        img = cv2.imread(join(str(config.IMAGE_FOLDER), filename))
        ax = fig.add_subplot(rows, columns, i)
        ax.set_title(filename + "_" + str("{0:.2f}".format(distance)))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    if forTask1and3:
        outputFolder = join(config.DATABASE_FOLDER, outputFolderParams + "_output")
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        plt.savefig(join(outputFolder, str(finalImageName) + ".png"))
    else:
        plt.savefig(join(config.DATABASE_FOLDER, outputFolderParams + ".png"))


def plot_output_term_weight_pairs(object_semantics_filename):
    filename = object_semantics_filename
    object_semantics_filename = join(config.DATABASE_FOLDER, 'Object_Semantics_' + object_semantics_filename + ".json")
    with open(object_semantics_filename, "r") as f:
        data = json.load(f)
    k = 1
    for key, value in data.items():
        dictionaryOfImageAndSematic = {}
        for i in range(len(value)):
            semantic, imagename = tuple(value[i])
            dictionaryOfImageAndSematic[imagename] = semantic
        progress(k,len(data.items()))
        plot(len(value), dictionaryOfImageAndSematic, filename, finalImageName=k)
        k = k + 1


def find_distance_2_vectors(vector1, vector2):
    # distance.euclidean(vector1, vector2)
    return np.linalg.norm(vector1 - vector2, axis=1)


def save_pickle(pickle_file_path, obj):
    file_path = os.path.join(pickle_file_path)
    filehandler = open(file_path, 'wb')
    pickle.dump(obj, filehandler)


def load_pickle(pickle_file_path):
    with open(pickle_file_path, "rb") as f:
        return pickle.load(f)

