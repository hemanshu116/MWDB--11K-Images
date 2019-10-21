import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from Main.helper import find_distance_2_vectors


class LDA_Reducer:
    def __init__(self, featureDescriptor, k):
        self.featureDescriptor = featureDescriptor
        self.k = k

    def reduceDimension(self):
        clf = LinearDiscriminantAnalysis(n_components=self.k)
        clf.fit_transform(self.featureDescriptor)
        principalDf = pd.DataFrame(data=clf)
        print(principalDf)
        return principalDf

    def inv_transform(self, data):
        return self.scaler.inverse_transform(self.lda.inverse_transform(data))

    def saveImageID(self, imageID):
        self.imageID = imageID

    def compute_threshold(self):
        reconstructed_normalized_feat_desc = self.lda.inverse_transform(self.objectLatentsSemantics)
        reconstructed_feat_desc = self.scaler.inverse_transform(reconstructed_normalized_feat_desc)
        reconstruction_err = find_distance_2_vectors(reconstructed_feat_desc, self.featureDescriptor)
        self.threshold = np.max(reconstruction_err)
