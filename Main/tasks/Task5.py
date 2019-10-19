import os
import json
import pandas as pd
import numpy as np

import Main.config as config
from Main.config import frTechniqueDict, fdTechniqueDict, flTechniqueDict
from Main.featureDescriptors.CM import CM
from Main.featureDescriptors.HOG import HOG
from Main.featureDescriptors.LBP import LBP
from Main.featureDescriptors.SIFT import SIFT
from Main.reducers.PCA_Reducer import PCA_Reducer
from Main.reducers.LDA_Reducer import LDA_Reducer
from Main.reducers.SVD_Reducer import SVD_Reducer
from Main.reducers.NMF_Reducer import NMF_Reducer



def startTask5():
    # Get inputs from usr
    inps = get_usr_input()
    feat_desc_ch = inps['feature']
    feat_redux_ch = inps['reduction']
    label_ch = inps['label']
    k = inps['k']

    # Fetch the latent semantic corresponding to the feat desc, redux teknik, label and k val
    latent_semantics = get_latent_semantics(int(feat_desc_ch), int(feat_redux_ch), int(label_ch), int(k))
    # Pretty print the latent semantic returned
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(latent_semantics)
    print('Shape of latent sem: ', np.shape(latent_semantics), 'type: ', type(latent_semantics))

    # Transform matrix

    # Inverse Transform matrix

    # Calc diff b/w I and I'

    # Fd max diff and std dev for threshold

    # Get query img from usr

    # Project query img into latent space


def get_usr_input():
    print('Choose from any one of the following Feature Models')
    print('1. Color Moments\n2. Local Binary Pattern\n3. Histogram of Oriented Gradients\n4. Scale-invarient Feature Transform')
    feat_ch = input('Enter your choice: ')

    print('Choose from any one of the following reducer technique: ')
    print('1. PCA\n2. LDA\n3. SVD\n4. NMF')
    redux_ch = input('Enter your choice: ')

    k = input('Enter k (number of latent semantics): ')

    print('Choose from any one of the following Labels: ')
    print('1. left-hand\n2. right-hand\n3. dorsal\n4. palmar\n5. with accessories\n6. without accesories\n7. Male\n8. Female')
    label_ch = input('Enter your choice: ')

    return {'feature': feat_ch, 'reduction': redux_ch, 'k': k, 'label': label_ch}


def get_latent_semantics(feat_desc_ch, feat_redux_ch, label_ch, k):
    meta_data = pd.read_csv(config.METADATA_FOLDER)  # Opens csv to dataframe

    # Get subset of imgs corresponding to the label
    aspect_of_hand = get_aspect_of_hand(label_ch)
    img_subset = meta_data.loc[meta_data['aspectOfHand'].str.contains(aspect_of_hand)]

    # Create feature descriptor for the img subset
    feat_vec = get_feature_vector(feat_desc_ch, img_subset)

    # Apply dimensionality reduction
    latent_sem = create_latent_semantics(feat_redux_ch, feat_vec, k)

    return latent_sem


def get_aspect_of_hand(label_ch):
    if (int(label_ch) == 1): return 'left'
    elif (int(label_ch) == 2): return 'right'
    elif (int(label_ch) == 3): return 'dorsal'
    elif (int(label_ch) == 4): return 'palmar'
    elif (int(label_ch) == 5): return '1'
    elif (int(label_ch) == 6): return '2'
    elif (int(label_ch) == 7): return 'male'
    elif (int(label_ch) == 8): return 'female'


def get_feature_vector(feat_desc_ch, img_subset):
    if (feat_desc_ch == 1):
        cm = CM()
        return cm.CMFeatureDescriptorForImageSubset(img_subset)
    elif (feat_desc_ch == 2):
        lbp = LBP()
        return lbp.LBPFeatureDescriptorForImageSubset(img_subset)
    elif (feat_desc_ch == 3):
        hog = HOG()
        return hog.HOGFeatureDescriptorForImageSubset(img_subset)
    elif (feat_desc_ch == 4):
        sift = SIFT()
        return sift.SIFTFeatureDescriptorForImageSubset(img_subset)


def create_latent_semantics(feature_redux_teknik_ch, feature_vector, k):
    if (feature_redux_teknik_ch == 1):
        return PCA_Reducer(feature_vector, k).reduceDimension()
    elif (feature_redux_teknik_ch == 2):
        return LDA_Reducer(feature_vector, k).reduceDimension()
    elif (feature_redux_teknik_ch == 3):
        return SVD_Reducer(feature_vector, k).reduceDimension()
    elif (feature_redux_teknik_ch == 4):
        return NMF_Reducer(feature_vector, k).reduceDimension()

