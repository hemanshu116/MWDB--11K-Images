import numpy as np

from Main.featureDescriptors.CM import CM
from Main.featureDescriptors.HOG import HOG
from Main.featureDescriptors.LBP import LBP
from Main.helper import getUserInput
from Main.reducers.LDA_Reducer import LDA_Reducer
from Main.reducers.NMF_Reducer import NMF_Reducer
from Main.reducers.PCA_Reducer import PCA_Reducer
from Main.reducers.SVD_Reducer import SVD_Reducer


def startTask1():
    inputs = getUserInput()
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
    featureVector = np.asarray(featureVector)
    k = int(k)
    if int(frTechnique) == 1:
        pca = PCA_Reducer(featureVector, k).reduceDimension()
    if int(frTechnique) == 2:
        lda = LDA_Reducer(featureVector, k).reduceDimension()
    if int(frTechnique) == 3:
        nmf = NMF_Reducer(featureVector, k).reduceDimension()
    if int(frTechnique) == 4:
        svd = SVD_Reducer(featureVector, k).reduceDimension()


# Uncomment to run task independently
startTask1()
