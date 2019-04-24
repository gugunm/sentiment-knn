from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class Classification:
    def __init__(self, k=3):
        # Create KNN classifier
        self.knn = KNeighborsClassifier(n_neighbors = k)

    def clf(self, X_train, y_train, X_test):
        # split dataset into train and test data
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)   
        # Fit the classifier to the data
        self.knn.fit(X_train,y_train)
        # predictions on the test data
        prediction = self.knn.predict(X_test)
        return prediction