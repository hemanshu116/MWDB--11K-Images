import json
import os

import numpy as np
import pickle

import Main.config as config
from Main.config import frTechniqueDict, fdTechniqueDict, flTechniqueDict
from Main.tasks.Task3 import startTask3
from Main.helper import findDistance, normalize_score, printMatch


def startTask4():
    inputTask2 = getUserInputForTask4()
    fdTechnique = inputTask2[0]
    flTechnique = inputTask2[1]
    frTechnique = inputTask2[2]
    k = inputTask2[5]
    imageId = inputTask2[3]
    m = int(inputTask2[4])
    with open(os.path.join(config.DATABASE_FOLDER, frTechniqueDict.get(fdTechnique) + "_" + fdTechniqueDict.get(frTechnique) + "_"
              + flTechniqueDict.get(flTechnique) + "_" + k), "rb") as f:
        reducerObject = pickle.load(f)
    latentFeatureDict = {}
    data = reducerObject.reduceDimension(reducerObject.featureDescriptor)

    i = 0
    for filename in reducerObject.imageID:
        latent = data.iloc[i][:]
        latentFeatureDict[filename] = latent
        i = i + 1

    selectedImage = latentFeatureDict[imageId]
    distanceList = findDistance(selectedImage, latentFeatureDict)
    printMatch(distanceList, m)


def getUserInputForTask4():
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
    print("Please select one of the labels")
    print("1. Left-hand")
    print("2. Right-hand")
    print("3. Dorsal")
    print("4. Palmer")
    print("5. With accessories")
    print("6. Without accessories")
    print("7. Male")
    print("8. Female")
    flInput = input()
    print("Please enter K number of latent semantics")
    k = input()
    fileExists = os.path.exists(config.DATABASE_FOLDER + frTechniqueDict.get(fdInput) + "_"
                                + fdTechniqueDict.get(frInput) + "_" + flTechniqueDict.get(flInput) + "_" + k)
    if fileExists:
        print("Database found, still want to recompute? (Y/N)")
        shouldRecompute = input()
        if shouldRecompute.lower() == "y":
            print("Please enter K number of latent semantics")
            k = input()
            startTask3([fdInput, flInput, frInput, k], False)
    else:
        print("Database was not found, Please enter k for computing")
        startTask3([fdInput, flInput, frInput, k], False)
    print("Enter the image path for matching")
    imagePath = input()
    print("Enter the number of matches to return")
    m = input()
    return [fdInput, flInput, frInput, imagePath, m, k]


# Uncomment to run task independently
# startTask4()
