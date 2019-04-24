from sklearn.model_selection import KFold

class CrossValidation:
    def __init__(self, k_split=5):
        # Splitter
        self.kfold = KFold(n_splits=k_split, shuffle=False)

    def splitter(self):
        return self.kfold
