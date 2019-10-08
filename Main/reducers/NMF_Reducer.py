import pandas as pd
from sklearn.decomposition import NMF


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
