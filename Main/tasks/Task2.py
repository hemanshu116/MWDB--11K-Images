import json
import os
import pickle
from os.path import join
from shutil import copyfile

import numpy as np
import pandas as pd

from Main import config
from Main.config import DATABASE_FOLDER, frTechniqueDict, fdTechniqueDict
from Main.helper import findDistance, normalize_score, printMatch
from Main.tasks.Task1 import startTask1


def startTask2():
    inputTask2 = getUserInputForTask2()
    fdTechnique = inputTask2[0]
    frTechnique = inputTask2[1]
    imageId = inputTask2[2]
    m = inputTask2[3]
    k = inputTask2[4]
    with open(join(config.DATABASE_FOLDER , frTechniqueDict[fdTechnique] + '_' + fdTechniqueDict[frTechnique] + "_" + str(k)),
              "rb") as f:
        reducerObject = pickle.load(f)
    latentFeatureDict = {}
    data = reducerObject.reduceDimension(reducerObject.featureDescriptor)
    i = 0
    for file in os.listdir(str(config.IMAGE_FOLDER)):
        filename = os.fsdecode(file)
        latent = data.iloc[i][:]
        latentFeatureDict[filename] = latent
        i = i + 1

    selectedImage = latentFeatureDict[imageId]
    distanceList = findDistance(selectedImage, latentFeatureDict)
    printMatch(distanceList, m, frTechniqueDict[fdTechnique] + '_' + fdTechniqueDict[frTechnique] + "_" + str(k))


def getUserInputForTask2():
    print("Please select the feature descriptor technique")
    print("1. Color Moments")
    print("2. Linear Binary Patterns")
    print("3. Histogram of Oriented Gradients")
    print("4. SIFT")
    fdInput = input()
    print("Please select the feature reduction technique")
    print("1. PCA")
    print("2. LDA")
    print("3. SVD")
    print("4. NMF")
    frInput = input()
    print("Please enter K number of latent semantics")
    k = input()
    fileExists = os.path.exists(
        join(DATABASE_FOLDER , frTechniqueDict.get(fdInput) + "_" + fdTechniqueDict.get(frInput) + "_" + str(k)))
    if fileExists:
        print("Database found, still want to recompute? (Y/N)")
        shouldRecompute = input()
        if shouldRecompute.lower() == "y":
            startTask1([fdInput, frInput, k], False)
    else:
        print("Database was not found, computing...")
        startTask1([fdInput, frInput, k], False)
    print("Enter the image path for matching")
    imagePath = input()
    print("Enter the number of matches to return")
    m = input()
    return [fdInput, frInput, imagePath, int(m), int(k)]


# Uncomment to run task independently
if __name__ == "__main__":
    startTask2()
