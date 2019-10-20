import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

from Main.helper import find_distance_2_vectors



class NMF_Reducer:
    def __init__(self, featureDescriptor, k):
        self.featureDescriptor = featureDescriptor
        self.k = k


    def reduceDimension(self):
        model = NMF(self.k, init='random', random_state=0)
        W = model.fit_transform(self.featureDescriptor)
        H = model.components_
        principalDf = pd.DataFrame(data=W)
        print(principalDf)
        return principalDf

    def inv_transform(self, data):
        return self.nmf.inverse_transform(data)


    def saveImageID(self, imageID):
        self.imageID = imageID


    def compute_threshold(self):
        reconstructed_feat_desc = self.nmf.inverse_transform(self.objectLatentsSemantics)
        threshold_list = find_distance_2_vectors(reconstructed_feat_desc, self.featureDescriptor)
        self.threshold = np.max(threshold_list)

