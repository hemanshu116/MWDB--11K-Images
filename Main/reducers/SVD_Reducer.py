import pandas as pd
import numpy as np


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

    def saveImageID(self, imageID):
        self.imageID = imageID