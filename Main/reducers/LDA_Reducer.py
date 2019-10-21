import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from Main.helper import find_distance_2_vectors
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler




class LDA_Reducer:
    def __init__(self, featureDescriptor, k):
        self.featureDescriptor = featureDescriptor
        self.k = k
        self.scaler = MinMaxScaler()
        self.scaler.fit(featureDescriptor)
        self.imageID = None
        self.model = LatentDirichletAllocation(self.k, random_state=0)
        W = self.model.fit_transform(self.scaler.transform(self.featureDescriptor))
        H = self.model.components_
        self.featureLatentSemantics = H[:self.k, :].T
        self.objectLatentSemantics = W[:, :self.k]

    def reduceDimension(self, data):
        reducedDimensions = self.model.transform(data)
        return pd.DataFrame(data=reducedDimensions)

    def inv_transform(self, data):
        return self.scaler.inverse_transform(self.model.inverse_transform(data))

    def saveImageID(self, imageID):
        self.imageID = imageID

    def compute_threshold(self):
        reconstructed_normalized_feat_desc = self.model.inverse_transform(self.objectLatentSemantics)
        reconstructed_feat_desc = self.scaler.inverse_transform(reconstructed_normalized_feat_desc)
        reconstruction_err = find_distance_2_vectors(reconstructed_feat_desc, self.featureDescriptor)
        self.threshold = np.max(reconstruction_err)
