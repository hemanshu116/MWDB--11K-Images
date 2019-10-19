import json
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

import Main.config as config
from Main.config import frTechniqueDict, fdTechniqueDict
from Main.featureDescriptors.CM import CM
from Main.featureDescriptors.HOG import HOG
from Main.featureDescriptors.LBP import LBP
from Main.reducers.LDA_Reducer import LDA_Reducer
from Main.reducers.NMF_Reducer import NMF_Reducer
from Main.reducers.PCA_Reducer import PCA_Reducer
from Main.reducers.SVD_Reducer import SVD_Reducer


def saveToFile(fr, frType, fdType):
    store = {}
    i = 0
    for file in os.listdir(str(config.IMAGE_FOLDER)):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            store[i] = filename
            i = i + 1

    fr.rename(index=store, inplace=True)
    # print(json.loads(fr.to_json()))
    json_file = os.path.join(config.DATABASE_FOLDER, frTechniqueDict[frType] + '_' + fdTechniqueDict[fdType] + '.json')
    with open( json_file, 'w', encoding='utf-8') as f:
        json.dump(json.loads(fr.to_json(orient='index')), f, ensure_ascii=True, indent=4)


def startTask1(inputs=[], shouldGetInputs=True):
    if shouldGetInputs:
        inputs = getUserInputForTask1()
    fdTechnique = inputs[0]
    frTechnique = inputs[1]
    k = inputs[2]

    featureVector = []

    if int(fdTechnique) == 1:
        cm = CM()
        featureVector = cm.CMFeatureDescriptor()
        # return featureVector
    elif int(fdTechnique) == 2:
        lbp = LBP()
        featureVector = lbp.LBPFeatureDescriptor()
        # return featureVector
    elif int(fdTechnique) == 3:
        hog = HOG()
        featureVector = hog.HOGFeatureDescriptor()
        # return featureVector
    elif int(fdTechnique) == 4:
        pass
    else:
        print("Wrong input")
        exit()

    k = int(k)
    fr = ""
    print('length of feature vector', len(featureVector))

    if int(frTechnique) == 1:
        fr = PCA_Reducer(featureVector, k)
    if int(frTechnique) == 2:
        fr = LDA_Reducer(featureVector, k)
    if int(frTechnique) == 3:
        fr = SVD_Reducer(featureVector, k)
    if int(frTechnique) == 4:
        fr = NMF_Reducer(featureVector, k)

    # save for visualization
    store = []
    for file in os.listdir(str(config.IMAGE_FOLDER)):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            store.append(filename)
    # plot_term_weight_pairs(fr.objectLatentsSemantics, store)

    filehandler = open(config.DATABASE_FOLDER + frTechniqueDict[fdTechnique] + '_' + fdTechniqueDict[frTechnique], 'wb')
    pickle.dump(fr, filehandler)


def getUserInputForTask1():
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
    return [fdInput, frInput, k]


def plot_term_weight_pairs(components, col_index):
    components_df = pd.DataFrame(components)
    print(components.shape)
    components_df['index'] = col_index
    components_df.set_index('index')
    print(components)
    output = {}
    num_components = len(components_df.columns)
    for i in range(0, num_components):
        sorted_vals = components_df.iloc[:, i].sort_values(ascending=False)
        output[i] = (list(zip(sorted_vals, sorted_vals.index)))
    fp = open(config.DATABASE_FOLDER + 'test.json', 'w')
    json.dump(output, fp)


# Uncomment to run task independently
# startTask1()
