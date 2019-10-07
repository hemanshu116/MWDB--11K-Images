import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA_Reducer:
    def __init__(self, featureDescriptor, k):
        self.featureDescriptor = featureDescriptor
        self.k = k

    def reduceDimension(self):
        clf = LinearDiscriminantAnalysis(n_components=self.k)
        clf.fit_transform(self.featureDescriptor)
        principalDf = pd.DataFrame(data=clf)
        print(principalDf)
