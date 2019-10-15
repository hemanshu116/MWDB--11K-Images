import pandas as pd
import numpy as np


class SVD_Reducer:
    def __init__(self, featureDescriptor, k):
        self.featureDescriptor = featureDescriptor
        self.k = k

    def reduceDimension(self):
        U, S, VT = np.linalg.svd(self.featureDescriptor, full_matrices=True)
        principalDf = pd.DataFrame(data=U[:, :self.k])
        return principalDf
