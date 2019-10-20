import os
import pandas as pd
import numpy as np

import Main.config as config
from Main.helper import load_pickle, find_distance_2_vectors
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

    # Fetch model trained for the corresponding feature model, redux technik, label and k
    json_file = frTechniqueDict.get(feat_desc_ch) + "_" + fdTechniqueDict.get(feat_redux_ch) + "_" + flTechniqueDict.get(label_ch) + "_" + k
    pickle_file_path = os.path.join(config.DATABASE_FOLDER, json_file)
    reducer_object = load_pickle(pickle_file_path)

    # get unlabelled img threshold
    unseen_img_desc = get_unseen_img_fd(int(feat_desc_ch))  # Get unseen img from usr and returns feature descriptor.
    unseen_img_threshold = get_unseen_img_threshold(unseen_img_desc, reducer_object)

    if (unseen_img_threshold > reducer_object.threshold):
        print('Not part of this label')
    else:
        print('Part of the label')


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

# Unused
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

# Unused
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

# Unused
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


# Unused
def create_latent_semantics(feature_redux_ch, feature_vector, k):
    if (feature_redux_ch == 1):
        return PCA_Reducer(feature_vector, k).reduceDimension()
    elif (feature_redux_ch == 2):
        return LDA_Reducer(feature_vector, k).reduceDimension()
    elif (feature_redux_ch == 3):
        return SVD_Reducer(feature_vector, k).reduceDimension()
    elif (feature_redux_ch == 4):
        return NMF_Reducer(feature_vector, k).reduceDimension()


def get_unseen_img_fd(feat_desc_ch):
    img_id = input('Enter image ID to be labelled: ')
    img_path = os.path.join(config.FULL_IMAGESET_FOLDER, img_id)

    if (feat_desc_ch == 1):
        cm = CM()
        return cm.CMForSingleImage(img_path)
    elif (feat_desc_ch == 2):
        lbp = LBP()
        return lbp.LBPForSingleImage(img_path)
    elif (feat_desc_ch == 3):
        hog = HOG()
        return hog.HOGForSingleImage(img_path)
    elif (feat_desc_ch == 4):
        sift = SIFT()
        return sift.SIFTForSingleImage(img_path)


def get_unseen_img_threshold(unseen_img_desc, reducer_obj):
    tmp = np.reshape(unseen_img_desc, (-1, len(unseen_img_desc)))   # Converting 1D arr to 2D arr.
    reduced_desc = reducer_obj.reduceDimension(tmp)
    reconstructed_desc = reducer_obj.inv_transform(reduced_desc)
    threshold = find_distance_2_vectors(unseen_img_desc, reconstructed_desc)
    return threshold[0]
