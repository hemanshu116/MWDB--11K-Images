import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

class SVD_Reducer:
    def __init__(self, featureDescriptor, k):
        self.featureDescriptor = featureDescriptor
        self.k = k

    def reduceDimension(self):
        U, S, VT = svds(self.featureDescriptor, self.k)
        # svd = TruncatedSVD(n_components=self.k)
        # VT = svd.fit(self.featureDescriptor)
        principalDf = pd.DataFrame(data=VT)     # Converting matrix to dataframe.
        return principalDf
