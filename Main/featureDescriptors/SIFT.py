import cv2
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing

import Main.config as config
from Main.helper import progress


class SIFT:

    def SIFTForSingleImage(self, filename):
        feature_df = SIFT.compute_features_by_file(filename, SIFT.compute_sift_features)
        return feature_df

    @classmethod
    def SIFTFeatureDescriptor(self):
        feature_df = SIFT.compute_features_by_folder(config.IMAGE_FOLDER, SIFT.compute_sift_features, return_gray=True)
        # print(feature_df)
        bow_list = SIFT.compute_BOVW(feature_df['Features'], 500)
        return np.array(bow_list)

    @classmethod
    def SIFTFeatureDescriptorForImageSubset(self, imageSet):
        feature_list = []
        number_files = len(imageSet)
        # print(imageSet)
        i = 1
        for fileName in imageSet:
            img = SIFT.load_img_by_id(join(config.IMAGE_FOLDER, fileName), True)
            features = SIFT.compute_sift_features(img)
            feature_list.append(features)
            progress(i, number_files)
            i = i + 1
        print()
        feature_df = pd.DataFrame({'FileName': imageSet, 'Features': feature_list})
        # print(feature_df)
        bow_list = SIFT.compute_BOVW(feature_df['Features'], 500)
        feature_df = pd.DataFrame(bow_list)
        return np.array(bow_list)

    # Function to compute sift features
    @staticmethod
    def compute_sift_features(img_gray):
        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.08, edgeThreshold=15)
        keyPoints, desc = sift.detectAndCompute(img_gray, None)
        return np.asarray(desc)

    # Function to execute a given feature extraction technique per single file.
    @staticmethod
    def compute_features_by_file(file_name, ftr_comp_func, return_gray=True):
        feature_list = []
        img = SIFT.load_img_by_id(file_name, return_gray)
        features = ftr_comp_func(img)
        feature_list.append(features)
        return pd.DataFrame({'FileName': [file_name], 'Features': feature_list})

    # Function to execute a given feature extraction technique per folder.
    @staticmethod
    def compute_features_by_folder(folder_name, ftr_comp_func, return_gray=True):
        folder_path = folder_name
        fileNames = [f for f in listdir(folder_path) if (isfile(join(folder_path, f)) and not (f.startswith('.')))]
        # print(fileNames)
        feature_list = []
        number_files = len(fileNames)
        i = 1
        for fileName in fileNames:
            # print folder_path
            img = SIFT.load_img_by_id(join(folder_path, fileName), return_gray)
            features = ftr_comp_func(img)
            feature_list.append(features)
            progress(i, number_files)
            i = i + 1
        print()
        return pd.DataFrame({'FileName': fileNames, 'Features': feature_list})

    # Function to load images.
    # Supports color image and Y channel
    @staticmethod
    def load_img_by_id(imgPath, return_gray=True):
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        if return_gray:
            return img[:, :, 0]
        return img

    # Function to Bag of Visual Words
    @staticmethod
    def compute_BOVW(feature_descriptors, n_clusters=100):
        print("Bag of visual words with clusters: ", n_clusters)
        # print(feature_descriptors.shape)

        combined_features = np.vstack(np.array(feature_descriptors))
        print("Size of stacked features: ", combined_features.shape)

        std_scaler = StandardScaler()
        combined_features = std_scaler.fit_transform(combined_features)

        print("Starting K-means training")
        kmeans = KMeans(n_clusters=n_clusters, random_state=777).fit(combined_features)

        print("Finished K-means training, moving on to prediction")
        bovw_vector = np.zeros([len(feature_descriptors), n_clusters])

        for index, features in enumerate(feature_descriptors):
            features_scaled = std_scaler.transform(features)
            for i in kmeans.predict(features_scaled):
                bovw_vector[index, i] += 1

        # bovw_vector_normalized = preprocessing.normalize(bovw_vector, norm='l2')

        print("Finished K-means")
        return list(bovw_vector)
