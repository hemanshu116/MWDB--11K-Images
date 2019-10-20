import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCA_Reducer:
    def __init__(self, featureDescriptor, k):
        self.featureDescriptor = featureDescriptor
        self.k = k
        self.imageID = None
        self.pca = PCA(n_components=self.k)
        self.scaler = StandardScaler()
        self.scaler.fit(self.featureDescriptor)
        self.normalizedFeatureDescriptor = self.scaler.transform(self.featureDescriptor)
        if min(self.normalizedFeatureDescriptor.shape) <= k:
            print("Cannot compute on PCA on components higher than min of", self.normalizedFeatureDescriptor.shape)
            exit()
        self.pca.fit(self.normalizedFeatureDescriptor)
        self.featureLatentSemantics = self.pca.components_.T
        self.objectLatentsSemantics = self.pca.transform(featureDescriptor)

    def reduceDimension(self, data):
        reducedDimesnions = self.pca.transform(self.scaler.transform(data))
        return pd.DataFrame(data=reducedDimesnions)

    def saveImageID(self, imageID):
        self.imageID = imageID
