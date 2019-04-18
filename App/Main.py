from featureExtraction import FeatureExtraction
from featureSelection import FeatureSelection
from preprocessing import Preprocessing


if __name__ == "__main__":
    data_path = 'data/input/*.csv'

    # --- Load Preprocessing ---
    print("=== Preprocessing ===")
    data = Preprocessing().cleandata(path=data_path, lemmatizer=True)

    # --- Handle Ambiguitas B.Inggris (negasi) ---

    # --- Cross Validation Using 10-Fold ---

    # --- Feature Extraction TF-IDF ---
    print("=== Feature Extraction ===")
    tfidf = FeatureExtraction().get_tf_idf(data["ulasan"])

    # --- Feature Selection C-Square ---
    print("=== Feature Selection ===")
    selection = FeatureSelection().featureselection(tfidf, data["label"])

    # --- Classification Using KNN ---
    print("=== Classification ===")
    

    # --- Evaluation Recall, Precission, F-Measure ---

    print(selection)
    print(selection.shape)