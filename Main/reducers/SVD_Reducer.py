import pandas as pd
import numpy as np

from Main.helper import find_distance_2_vectors



class SVD_Reducer:
    def __init__(self, featureDescriptor, k):
        self.featureDescriptor = featureDescriptor
        self.k = k
        self.imageID = None
        U, S, VT = np.linalg.svd(self.featureDescriptor, full_matrices=True)
        self.featureLatentSemantics = VT[:self.k, :].T
        self.objectLatentsSemantics = U[:, :self.k]

    def reduceDimension(self, featureDescriptor):
        # self.U, self.S, self.VT = np.linalg.svd(self.featureDescriptor, full_matrices=False)
        principalDf = pd.DataFrame(data=np.dot(featureDescriptor, self.featureLatentSemantics))
        return principalDf

    def inv_transform(self, data):
        return np.dot(data, np.transpose(self.featureLatentSemantics))

    def saveImageID(self, imageID):
        self.imageID = imageID

    def compute_threshold(self):
        Z = np.dot(self.objectLatentsSemantics, self.featureLatentSemantics)
        reconstructed_feat_desc = np.dot(Z, np.transpose(self.featureLatentSemantics))
        reconstruction_err = find_distance_2_vectors(reconstructed_feat_desc, self.featureDescriptor)
        self.threshold = np.max(reconstruction_err)

