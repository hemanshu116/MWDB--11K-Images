import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from Main.helper import find_distance_2_vectors


class PCA_Reducer:
    def __init__(self, featureDescriptor, k):
        self.featureDescriptor = featureDescriptor
        self.k = k
        self.imageID = None
        self.pca = PCA(n_components=self.k)
        self.pca.fit(featureDescriptor)
        self.featureLatentSemantics = self.pca.components_.T
        # doubt???
        self.objectLatentsSemantics = self.pca.transform(featureDescriptor)

    def reduceDimension(self, data):
        reducedDimesnions = self.pca.transform(data)
        return pd.DataFrame(data=reducedDimesnions)

    def inv_transform(self, data):
        return self.pca.inverse_transform(data)

    def saveImageID(self, imageID):
        self.imageID = imageID

    def compute_threshold(self):
        reconstructed_feat_desc = self.inv_transform(self.objectLatentsSemantics)
        threshold_list = find_distance_2_vectors(reconstructed_feat_desc, self.featureDescriptor)
        self.threshold = np.max(threshold_list)

