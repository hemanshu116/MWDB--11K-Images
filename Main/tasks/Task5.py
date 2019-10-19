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

    # Fetch feature vector and latent semantic corresponding to the feat desc, redux teknik, label and k val
    feature_vec, latent_semantics = get_latent_semantics(int(feat_desc_ch), int(feat_redux_ch), int(label_ch), int(k))


    # Project sample imgs into the latent space
    img_latent_features = np.dot(feature_vec, np.transpose(latent_semantics))   # Will this work for all feat desc?

    # Inverse Transform matrix
    # feature_vec_reconstructed =

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
    # Get subset of imgs corresponding to the label
    img_subset = get_label_match_imgs(label_ch)
    print('Img Subset: ', np.shape(img_subset), type(img_subset))
    print(img_subset.head())

    # Create feature descriptor for the img subset
    feat_vec = get_feature_vector(feat_desc_ch, img_subset)
    print('Feature vector: ', np.shape(feat_vec), type(feat_vec))

    # Apply dimensionality reduction
    latent_sem = create_latent_semantics(feat_redux_ch, feat_vec, k)
    # Pretty print the latent semantic returned
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(latent_semantics)
    print('Shape of latent sem: ', np.shape(latent_sem), 'type: ', type(latent_sem))
    print(latent_sem.tail(), latent_sem.head())

    return feat_vec, latent_sem


def get_label_match_imgs(label_ch):
    meta_data = pd.read_csv(config.METADATA_FOLDER)  # Opens csv to dataframe


    if (int(label_ch) == 1):
        return meta_data.loc[meta_data['aspectOfHand'].str.contains('left')]
    elif (int(label_ch) == 2):
        return meta_data.loc[meta_data['aspectOfHand'].str.contains('right')]
    elif (int(label_ch) == 3):
        return meta_data.loc[meta_data['aspectOfHand'].str.contains('dorsal')]
    elif (int(label_ch) == 4):
        return meta_data.loc[meta_data['aspectOfHand'].str.contains('palmar')]
    elif (int(label_ch) == 5):
        return meta_data.loc[meta_data['accessories'].str.contains('1')]
    elif (int(label_ch) == 6):
        return meta_data.loc[meta_data['accessories'].str.contains('0')]
    elif (int(label_ch) == 7):
        return meta_data.loc[meta_data['gender'].str.contains('male')]
    elif (int(label_ch) == 8):
        return meta_data.loc[meta_data['gender'].str.contains('female')]


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


def create_latent_semantics(feature_redux_ch, feature_vector, k):
    if (feature_redux_ch == 1):
        return PCA_Reducer(feature_vector, k).reduceDimension()
    elif (feature_redux_ch == 2):
        return LDA_Reducer(feature_vector, k).reduceDimension()
    elif (feature_redux_ch == 3):
        return SVD_Reducer(feature_vector, k).reduceDimension()
    elif (feature_redux_ch == 4):
        return NMF_Reducer(feature_vector, k).reduceDimension()

