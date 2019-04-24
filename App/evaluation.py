from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Evaluation:
    def __init__(self):
        pass

    def accuracy(self, y_test, y_pred):
        # model accuracy for x_test  
        precision = precision_score(y_test, y_pred, average="macro")
        recall    = recall_score(y_test, y_pred, average="macro")
        f_measure = f1_score(y_test, y_pred, average="macro")
        accuracy  = accuracy_score(y_test, y_pred)

        return(precision, recall, f_measure, accuracy)