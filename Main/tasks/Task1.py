import json
import os
import pickle
from os.path import join

import pandas as pd

import Main.config as config
from Main.config import frTechniqueDict, fdTechniqueDict
from Main.featureDescriptors.CM import CM
from Main.featureDescriptors.HOG import HOG
from Main.featureDescriptors.LBP import LBP
from Main.featureDescriptors.SIFT import SIFT
from Main.helper import plot_output_term_weight_pairs
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
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json.loads(fr.to_json(orient='index')), f, ensure_ascii=True, indent=4)


def startTask1(inputs=[], shouldGetInputs=True):
    if shouldGetInputs:
        inputs = getUserInputForTask1()
    fdTechnique = inputs[0]
    frTechnique = inputs[1]
    k = inputs[2]

    featureVector = []
    fileExists = os.path.exists(join(config.DATABASE_FOLDER, frTechniqueDict.get(fdTechnique) + "_"
                                     + fdTechniqueDict.get(frTechnique)+ "_" + k))
    if(fileExists):
        output_filename = frTechniqueDict[fdTechnique] + '_' + fdTechniqueDict[frTechnique] + "_" + str(k)
        plot_output_term_weight_pairs(output_filename)
        exit()
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
        sift = SIFT()
        featureVector = sift.SIFTFeatureDescriptor()
    else:
        print("Wrong input")
        exit()

    k = int(k)
    fr = ""

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

    output_filename = frTechniqueDict[fdTechnique] + '_' + fdTechniqueDict[frTechnique] + "_" + str(k)

    output_term_weight_pairs(fr.objectLatentSemantics, store,
                             join(config.DATABASE_FOLDER, 'Object_Semantics_' + output_filename))

    print(fr.featureLatentSemantics.shape)
    output_term_weight_pairs(fr.featureLatentSemantics,
                             ['f' + str(x) for x in range(0, len(fr.featureLatentSemantics))],
                             join(config.DATABASE_FOLDER, 'Feature_Semantics_' + output_filename))

    filehandler = open(join(config.DATABASE_FOLDER, output_filename), 'wb')
    print("Term weight pairs are successfully stored")
    print("progress for visualization..")
    plot_output_term_weight_pairs(output_filename)
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


def output_term_weight_pairs(components, col_index, filepath):
    components_df = pd.DataFrame(components).T
    # print(components.shape)
    components_df.columns = col_index
    output = {}
    num_components = len(components_df)
    for i in range(1, num_components + 1):
        if i > 3:
            break
        sorted_vals = components_df.iloc[i - 1, :].sort_values(ascending=False)
        sorted_vals = sorted_vals.head(5)
        output[i] = (list(zip(sorted_vals, sorted_vals.index)))
    fp = open(filepath + '.json', 'w+')
    json.dump(output, fp)


# Uncomment to run task independently
if __name__ == "__main__":
    startTask1()
