from Main.featureDescriptors.CM import CM
from Main.featureDescriptors.HOG import HOG
from Main.featureDescriptors.LBP import LBP


def getUserInput():
    print("Please select the image path")
    userInput = input()
    print("Please select the feature descriptor technique")
    print("1. Color Moments")
    print("2. Linear Binary Patterns")
    print("3. Histogram of Oriented Gradients")
    print("4. SIFT")
    fdInput = input()
    print("Please select the feature reduction technique")
    print("1. PDA")
    print("2. LDA")
    print("3. SVD")
    print("4. NMF")
    frInput = input()
    return [userInput, fdInput, frInput]


def processFeatureDescriptors():
    inputs = getUserInput()
    imagePath = inputs[0]
    fdTechnique = inputs[1]
    frTechnique = inputs[2]

    featureVector = []

    if int(fdTechnique) == 1:
        cm = CM(imagePath)
        featureVector = cm.calculateFeatureDescriptor()
        return featureVector
    elif int(fdTechnique) == 2:
        lbp = LBP(imagePath)
        featureVector = lbp.calculateFeatureDiscriptor()
        return featureVector
    elif int(fdTechnique) == 3:
        hog = HOG(imagePath)
        featureVector = hog.calculateFeatureDiscriptor()
        return featureVector
    elif int(fdTechnique) == 4:
        pass
    else:
        print("Wrong input")
        exit()
