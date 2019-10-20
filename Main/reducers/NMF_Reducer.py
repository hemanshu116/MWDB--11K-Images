import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler


class NMF_Reducer:
    def __init__(self, featureDescriptor, k):
        self.featureDescriptor = featureDescriptor
        self.k = k
        scaler = MinMaxScaler()
        scaler.fit(featureDescriptor)
        if min(len(featureDescriptor[0]), len(featureDescriptor)) <= k:
            print()
            print("Cannot compute on NMF on components higher than min of [", len(featureDescriptor[0]), ",",
                  len(featureDescriptor)," ]")
            exit()
        self.model = NMF(self.k, init='random', random_state=0)
        W = self.model.fit_transform(scaler.transform(self.featureDescriptor))
        H = self.model.components_
        self.featureLatentSemantics = H[:self.k, :].T
        self.objectLatentsSemantics = W[:, :self.k]

    def reduceDimension(self, data):
        reducedDimesnions = self.transform(data)
        return pd.DataFrame(data=reducedDimesnions)
