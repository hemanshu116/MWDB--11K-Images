import pandas as pd

import Main.config as config

def startTask3(inputs=[], shouldGetInputs=True):
    if shouldGetInputs:
        inputs = getUserInputForTask1()
    fdTechnique = inputs[0]
    flTechnique = inputs[1]
    frTechnique = inputs[2]
    k = inputs[3]
    meta_data = pd.read_csv(config.METADATA_FOLDER)
    aspectOfHand = ''
    accessories = ''
    gender = ''
    if flTechnique[1] == '1':
        aspectOfHand += 'dorsal'
    elif flTechnique[1] == '2':
        aspectOfHand += 'palmer'
    else:
        print("Wrong input")
        exit()
    if flTechnique[0] == '1':
        aspectOfHand += 'left'
    elif flTechnique[0] == '2':
        aspectOfHand += 'right'
    else:
        print("Wrong input")
        exit()
    if flTechnique[2] == '1':
        accessories = 1
    elif flTechnique[2] == '2':
        accessories = 0
    else:
        print("Wrong input")
        exit()
    if flTechnique[3] == '1':
        gender = 'male'
    elif flTechnique[3] == '2':
        gender = 'female'
    else:
        print("Wrong input")
        exit()
    featureVector = meta_data.loc[(meta_data['aspectOfHand'] == aspectOfHand) & (meta_data['accessories'] == accessories) &
                                  (meta_data['gender'] == gender)]




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
    fl1Input = input()
    print("1. Dorsal")
    print("2. Palmer")
    fl2Input = input()
    print("1. With accessories")
    print("2. Without accessories")
    fl3input = input()
    print("1. Male")
    print("2. Female")
    fl4Input = input()
    print("Please select the feature reduction technique")
    print("1. PCA")
    print("2. LDA")
    print("3. SVD")
    print("4. NMF")
    frInput = input()
    print("Please enter K number of latent semantics")
    k = input()
    return [fdInput, [fl1Input, fl2Input, fl3input, fl4Input], frInput, k]