# import cv2
import cv2
import numpy as np
import pandas as pd
from scipy.stats.mstats import skew
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import skew
from sklearn import preprocessing


class SIFT:
    def __init__(self, fileName):
        self.fileName = fileName

    @staticmethod
    def load_img_by_id(imgPath, return_gray=True):
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        if return_gray:
            return img[:, :, 0]
        return img

    def compute_sift_features(self,img_gray):
        print("came")
        # sift = cv2.xfeatures2d.SIFT_create()
        sift = cv2.xfeatures2d.SIFT_create()
        keyPoints, desc = sift.detectAndCompute(img_gray, None)
        return np.asarray(desc)

    def compute_features_by_file(self):
        feature_list = []
        fileNames = [self.fileName]
        print(self.fileName)
        img = self.load_img_by_id(self.fileName,True)
        features = self.compute_sift_features(img)
        feature_list.append(features)
        # print(pd.DataFrame({'FileName': fileNames, 'Features': feature_list}))
        return pd.DataFrame({'FileName': fileNames, 'Features': feature_list})

    def calculateFeatureDescriptor(self):
        pass


print(SIFT("D:\ASU Projects\CSE 551 - MWDB\Phase-1\CSE 515 Fall19 - Smaller Dataset\Hand_0008110.jpg").compute_features_by_file())
