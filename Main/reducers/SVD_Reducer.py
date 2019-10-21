import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from Main.helper import find_distance_2_vectors



class SVD_Reducer:
    def __init__(self, featureDescriptor, k):
        self.featureDescriptor = featureDescriptor
        self.k = k
        self.imageID = None
        self.scaler = StandardScaler()
        self.scaler.fit(self.featureDescriptor)
        U, S, VT = np.linalg.svd(self.scaler.transform(self.featureDescriptor), full_matrices=True)
        if min(U.shape) < k or min(VT.shape) < k:
            print("Cannot compute on SVD on components higher than min of", U.shape)
            exit()
        self.featureLatentSemantics = VT[:self.k, :].T
        self.objectLatentSemantics = U[:, :self.k]

    def reduceDimension(self, featureDescriptor):
        principalDf = pd.DataFrame(data=np.dot(self.scaler.transform(featureDescriptor), self.featureLatentSemantics))
        return principalDf

    def inv_transform(self, data):
        return self.inv_transform(np.dot(data, np.transpose(self.featureLatentSemantics)))

    def saveImageID(self, imageID):
        self.imageID = imageID


    def compute_threshold(self):
        reconstructed_normalized_feat_desc = np.dot(self.objectLatentSemantics, np.transpose(self.featureLatentSemantics))
        # reconstructed_normalized_feat_desc = np.dot(Z, (self.featureLatentSemantics))
        reconstructed_feat_desc = self.scaler.inverse_transform(reconstructed_normalized_feat_desc)
        reconstruction_err = find_distance_2_vectors(reconstructed_feat_desc, self.featureDescriptor)
        print(np.shape(reconstruction_err), np.average(reconstruction_err))
        self.threshold = np.max(reconstruction_err)

