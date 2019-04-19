from sklearn.metrics import accuracy_score

class Evaluation:
    def __init__(self):
        pass

    def accuracy(self, y_test, y_pred):
        # model accuracy for x_test  
        accuracy = accuracy_score(y_test, y_pred)
        
        return(accuracy)