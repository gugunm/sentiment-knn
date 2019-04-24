from featureExtraction import FeatureExtraction
from featureSelection import FeatureSelection
from crossValidation import CrossValidation
from classification import Classification
from preprocessing import Preprocessing
from evaluation import Evaluation

if __name__ == "__main__":
    data_path = 'data/input/*.csv'

    # --- Load Preprocessing ---
    print("=== Preprocessing ===")
    data = Preprocessing().cleandata(path=data_path, lemmatizer=True)

    ulasan = data["ulasan"]
    label  = data["label"]

    # --- Handle Ambiguitas B.Inggris (negasi) ---

    # --- Feature Extraction TF-IDF ---
    print("=== Feature Extraction ===")
    tfidf = FeatureExtraction().get_tf_idf(ulasan)

    # --- Feature Selection C-Square ---
    print("=== Feature Selection === \n")
    selection = FeatureSelection().featureselection(tfidf, label)

    # --- Cross Validation Using K-Fold ---
    kfold = CrossValidation(k_split=5).splitter()
    count = 0
    for train_index, test_index in kfold.split(ulasan):
        count += 1
        print("=== Fold : {} ===".format(count))
        X_train, X_test = selection[train_index], selection[test_index] 
        y_train, y_test = label[train_index], label[test_index]

        # --- Classification Using KNN ---
        print("=== Classification ===")
        y_prediction = Classification(k=7).clf(X_train, y_train, X_test)

        # --- Evaluation Recall, Precission, F-Measure ---
        print("=== Evaluation ===")
        precision, recall, f_measure, accuracy  = Evaluation().accuracy(y_test, y_prediction)

        print('Precision : {0:.2f}% '.format(100 * precision))
        print('Recall : {0:.2f}% '.format(100 * recall))
        print('F-Measure : {0:.2f}% '.format(100 * f_measure))
        print('Accuracy : {0:.2f}% \n'.format(100 * accuracy))