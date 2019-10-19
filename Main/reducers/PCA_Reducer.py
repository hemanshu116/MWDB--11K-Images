import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


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

    def saveImageID(self, imageID):
        self.imageID = imageID