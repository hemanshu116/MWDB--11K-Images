import pandas as pd
from sklearn.decomposition import PCA


class PCA_Reducer:
    def __init__(self, featureDescriptor, k):
        self.featureDescriptor = featureDescriptor
        self.k = k

    def reduceDimension(self):
        pca = PCA(n_components=self.k)
        # print(type(self.featureDiscriptor))
        principalComponents = pca.fit_transform(self.featureDescriptor)
        # print(principalComponents)
        principalDf = pd.DataFrame(data=principalComponents)
        return principalDf
