import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCA_Reducer:
    def __init__(self, featureDescriptor, k):
        self.featureDescriptor = featureDescriptor
        self.k = k

    def reduceDimension(self):
        # Standardizing feature descriptors by making mean 0 and std deviation 1.
        print('Feat desc shape: ', (self.featureDescriptor))
        self.featureDescriptor = StandardScaler().fit_transform(self.featureDescriptor)
        pca = PCA(n_components=self.k)
        print('feat desc mean: ', np.mean(self.featureDescriptor), 'feat desc std dec: ',  np.std(self.featureDescriptor))
        principalComponents = pca.fit_transform(self.featureDescriptor)
        # print(principalComponents)
        principalDf = pd.DataFrame(data=principalComponents)
        return principalDf
