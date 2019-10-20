import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler


class LDA_Reducer:
    def __init__(self, featureDescriptor, k):
        self.featureDescriptor = featureDescriptor
        self.k = k
        scaler = MinMaxScaler()
        scaler.fit(featureDescriptor)
        self.imageID = None
        self.model = LatentDirichletAllocation(self.k, random_state=0)
        W = self.model.fit_transform(scaler.transform(self.featureDescriptor))
        H = self.model.components_
        self.featureLatentSemantics = H[:self.k, :].T
        self.objectLatentSemantics = W[:, :self.k]

    def reduceDimension(self, data):
        reducedDimensions = self.model.transform(data)
        return pd.DataFrame(data=reducedDimensions)

    def saveImageID(self, imageID):
        self.imageID = imageID