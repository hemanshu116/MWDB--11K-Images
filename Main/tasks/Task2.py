import json
import os
import pickle

import numpy as np
import pandas as pd

from Main import config
from Main.config import DATABASE_FOLDER, frTechniqueDict, fdTechniqueDict
from Main.helper import findDistance
from Main.tasks.Task1 import startTask1


def startTask2():
    inputTask2 = getUserInputForTask2()
    fdTechnique = inputTask2[0]
    frTechnique = inputTask2[1]
    imageId = inputTask2[2]
    m = inputTask2[3]
    with open(config.DATABASE_FOLDER + frTechniqueDict[fdTechnique] + '_' + fdTechniqueDict[frTechnique], "rb") as f:
        reducerObject = pickle.load(f)
    latentFeatureDict = {}
    data = reducerObject.reduceDimension(reducerObject.featureDescriptor)
    i=0
    for file in os.listdir(str(config.IMAGE_FOLDER)):
        filename = os.fsdecode(file)
        latent = data.iloc[i][:]
        latentFeatureDict[filename] = latent
        i=i+1

    selectedImage = latentFeatureDict[imageId]
    distanceList = findDistance(selectedImage, latentFeatureDict)
    printMatch(distanceList, m)


def printMatch(finalList, k):
    sortList = sorted(finalList.items(), key=lambda x: x[1])
    i = 0
    for keyValue in sortList:
        if i == int(k):
            break
        image, score = keyValue
        print(image + " : " + str(100.0 - score) + " % match")
        i = i + 1


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
    fileExists = os.path.exists(DATABASE_FOLDER + "\\" + frTechniqueDict.get(fdInput) + "_" + fdTechniqueDict.get(frInput))
    if fileExists:
        print("Database found, still want to recompute? (Y/N)")
        shouldRecompute = input()
        if shouldRecompute.lower() == "y":
            print("Please enter K number of latent semantics")
            k = input()
            startTask1([fdInput, frInput, k], False)
    else:
        print("Database was not found, Please enter k for computing")
        print("Please enter K number of latent semantics")
        k = input()
        startTask1([fdInput, frInput, k], False)
    print("Enter the image path for matching")
    imagePath = input()
    print("Enter the number of matches to return")
    m = input()
    return [fdInput, frInput, imagePath, int(m)]

# Uncomment to run task independently
# startTask2()
