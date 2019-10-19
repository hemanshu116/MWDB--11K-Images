import json
import os

import pandas as pd
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


def saveToFile(fr, frType, fdType, flabel, k):
    store = {}
    i = 0
    for file in os.listdir(str(config.IMAGE_FOLDER)):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            store[i] = filename
            i = i + 1

    fr.rename(index=store, inplace=True)
    # print(json.loads(fr.to_json()))
    json_file = frTechniqueDict[frType] + '_' + fdTechniqueDict[fdType] + '_' + flabel + '_' + k + '.json'
    with open(os.path.join(config.DATABASE_FOLDER, json_file), 'w', encoding='utf-8') as f:
        json.dump(json.loads(fr.to_json(orient='index')), f, ensure_ascii=True, indent=4)


def startTask3(inputs=[], shouldGetInputs=True):
    if shouldGetInputs:
        inputs = getUserInputForTask1()
    fdTechnique = inputs[0]
    flTechnique = inputs[1]
    frTechnique = inputs[2]
    k = int(inputs[3])

    meta_data = pd.read_csv(config.METADATA_FOLDER)
    if flTechnique == '1':
        aspectOfHand = 'left'
        flabel = 'LEFT'
        imageSet = meta_data.loc[(meta_data['aspectOfHand'].str.contains(aspectOfHand))]
    elif flTechnique == '2':
        aspectOfHand = 'right'
        flabel = 'RIGHT'
        imageSet = meta_data.loc[(meta_data['aspectOfHand'].str.contains(aspectOfHand))]
    elif flTechnique == '3':
        aspectOfHand = 'dorsal'
        flabel = 'DORSAL'
        imageSet = meta_data.loc[(meta_data['aspectOfHand'].str.contains(aspectOfHand))]
    elif flTechnique == '4':
        aspectOfHand = 'palmar'
        flabel = 'PALMAR'
        imageSet = meta_data.loc[(meta_data['aspectOfHand'].str.contains(aspectOfHand))]
    elif flTechnique == '5':
        accessories = '1'
        flabel = 'ACCESS'
        imageSet = meta_data.loc[(meta_data['accessories'].str.contains(accessories))]
    elif flTechnique == '6':
        accessories = '0'
        flabel = 'NOACCESS'
        imageSet = meta_data.loc[(meta_data['accessories'].str.contains(accessories))]
    elif flTechnique == '7':
        gender = 'male'
        flabel = 'MALE'
        imageSet = meta_data.loc[(meta_data['gender'].str.contains(gender))]
    elif flTechnique == '8':
        gender = 'female'
        flabel = 'FEMALE'
        imageSet = meta_data.loc[(meta_data['gender'].str.contains(gender))]
    else:
        print("Wrong input")
        exit()

    # print('Image set: ', imageSet)
    if fdTechnique == "1":
        cm = CM()
        featureVector = cm.CMFeatureDescriptorForImageSubset(imageSet)

    elif fdTechnique == "2":
        lbp = LBP()
        featureVector = lbp.LBPFeatureDescriptorForImageSubset(imageSet)

    elif fdTechnique == "3":
        hog = HOG()
        featureVector = hog.HOGFeatureDescriptorForImageSubset(imageSet)

    elif fdTechnique == "4":
        pass
    else:
        print("Wrong input")
        exit()


    if frTechnique == "1":
            fr = PCA_Reducer(featureVector, k).reduceDimension()
    if frTechnique == "2":
        fr = LDA_Reducer(featureVector, k).reduceDimension()
    if frTechnique == "3":
        fr = SVD_Reducer(featureVector, k).reduceDimension()
    if frTechnique == "4":
        fr = NMF_Reducer(featureVector, k).reduceDimension()
    saveToFile(fr, fdTechnique, frTechnique, flabel, str(k))


def getUserInputForTask1():
    print("Please select the feature descriptor technique")
    print("1. Color Moments")
    print("2. Linear Binary Patterns")
    print("3. Histogram of Oriented Gradients")
    print("4. SIFT")
    fdInput = input()

    print("Please select one of the labels")
    print("1. Left-hand")
    print("2. Right-hand")
    print("3. Dorsal")
    print("4. Palmar")
    print("5. With accessories")
    print("6. Without accessories")
    print("7. Male")
    print("8. Female")
    flInput = input()

    print("Please select the feature reduction technique")
    print("1. PCA")
    print("2. LDA")
    print("3. SVD")
    print("4. NMF")
    frInput = input()

    print("Please enter K number of latent semantics")
    k = input()

    return [fdInput, flInput, frInput, k]


# Uncomment to run task independently
# startTask3()