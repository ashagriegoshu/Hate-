import os
import numpy as np
from flask import Flask, request, render_template
import joblib
from scipy.sparse import hstack, csr_matrix

# ==========================
# Paths (adjusted to new saved_models layout)
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

MODEL_PATH = os.path.join(MODELS_DIR, "hate_speech_model.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
SVD_PATH = os.path.join(MODELS_DIR, "svd_reducer.pkl")

# ==========================
# Load models ONCE
# ==========================
# fail early with helpful message if missing
for p in (MODEL_PATH, TFIDF_PATH, SVD_PATH):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required model file not found: {p}. "
                                "Run model.py to train/create saved_models first.")

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
svd = joblib.load(SVD_PATH)

# ==========================
# Label Map
# ==========================
LABEL_MAP = {
    0: "Non-Hate Speech | ሽክትንዙ ዊግትንድ አየውም",
    1: "Race Hate Speech | ዚረዝ ሽክትንዙ ዊግትንድ",
    2: "Religion Hate Speech | ሀይማኖትዝ ሽክትንዙ ዊግትንድ",
    3: "Gender Hate Speech | ጾተዝ ሽክትንዙ ዊግትንድ",
    4: "Political Hate Speech | ፖለቲከዝ ሽክትንዙ ዊግትንድ"
}

# ==========================
# Flask App
# ==========================
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")
    if not text:
        return render_template("index.html", prediction_text="Please enter text to classify.", input_text="")

    # TF-IDF features (sparse)
    X_tfidf = tfidf.transform([text])  # shape (1, n_tfidf)

    # SVD dense embedding from TF-IDF
    X_svd = svd.transform(X_tfidf)     # shape (1, n_components)
    X_svd_sparse = csr_matrix(X_svd)   # convert to sparse for hstack

    # Combine (sparse + sparse)
    X_combined = hstack([X_tfidf, X_svd_sparse])

    # Predict
    pred = model.predict(X_combined)[0]
    result = LABEL_MAP.get(pred, f"Unknown label: {pred}")

    return render_template("index.html", prediction_text=result, input_text=text)

if __name__ == "__main__":
    # Change host/port as needed
    app.run(debug=True)