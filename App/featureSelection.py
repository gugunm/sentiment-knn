from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np

class FeatureSelection:
    def __init__(self):
        # Select K-best
        self.k = 5000
        self.chi2 = SelectKBest(score_func=chi2, k=self.k)

    def featureselection(self, X, Y):
        # test the model
        fit = self.chi2.fit(X, Y)
        features = fit.transform(X)
        return features