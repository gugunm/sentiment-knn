import Preprocessing as pr

if __name__ == "__main__":
    data_path = 'data/input/*.csv'

    # --- Load Preprocessing ---
    print("=== Preprocessing ===")
    data = pr.preprocessing(data_path=data_path)

    # --- Handle Ambiguitas B.Inggris (negasi) ---

    # --- Cross Validation Using 10-Fold ---

    # --- Feature Extraction TF-IDF ---

    # --- Feature Selection C-Square ---

    # --- Classification Using KNN ---

    # --- Evaluation Recall, Precission, F-Measure ---

    print(data)