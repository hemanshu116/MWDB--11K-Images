from Main.featureDescriptors.CM import CM
from Main.featureDescriptors.HOG import HOG
from Main.featureDescriptors.LBP import LBP
import numpy as np

from Main.reducers.PCA_Reducer import PCA_Reducer


def getUserInput():
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
