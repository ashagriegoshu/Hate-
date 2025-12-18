"""
COMPLETE HATE SPEECH DETECTION PIPELINE
WITH MODEL PERSISTENCE
Compatible with Python 3.14 (no gensim/C extensions)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# =========================
# PATHS & CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

PATHS = {
    'data': os.path.join(BASE_DIR, "geez_clean.csv"),
    'model': os.path.join(MODELS_DIR, "hate_speech_model.pkl"),
    'tfidf': os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"),
    'svd': os.path.join(MODELS_DIR, "svd_reducer.pkl"),
    'config': os.path.join(MODELS_DIR, "model_config.json")
}

# =========================
# TRAIN & SAVE MODELS (Run Once)
# =========================
def train_and_save_models(svd_components=100):
    """Train TF-IDF + SVD + classifier and save them for future use"""
    print("\n" + "="*60)
    print("TRAINING AND SAVING MODELS")
    print("="*60)
    
    # 1. Load data
    print("1. Loading dataset...")
    data = pd.read_csv(PATHS['data'], encoding="utf-8-sig")
    data = data.groupby("Label").filter(lambda x: len(x) > 1)
    X = data["Clean_Text"].astype(str)
    y = data["Label"]
    
    # 2. Train TF-IDF
    print("2. Training TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        analyzer="word",
        lowercase=False,
        min_df=2,
        max_df=0.95
    )
    X_tfidf = tfidf.fit_transform(X)
    
    # Save TF-IDF
    joblib.dump(tfidf, PATHS['tfidf'])
    print(f"   ✓ TF-IDF saved: {PATHS['tfidf']}")
    print(f"   Vocabulary size: {len(tfidf.vocabulary_)}")
    
    # 3. Train SVD (dense embedding over TF-IDF)
    print("3. Training TruncatedSVD (dense embedding) ...")
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)  # dense numpy array, shape (n_samples, svd_components)
    
    # Save SVD
    joblib.dump(svd, PATHS['svd'])
    print(f"   ✓ SVD saved: {PATHS['svd']}")
    print(f"   SVD components: {svd_components}")
    
    # 4. Create combined features
    print("4. Creating combined features...")
    X_svd_sparse = csr_matrix(X_svd)
    X_combined = hstack([X_tfidf, X_svd_sparse])
    
    # 5. Split data
    print("5. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Train classifier
    print("6. Training classifier...")
    model = LinearSVC(
        C=1.0,
        max_iter=10000,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # 7. Evaluate
    print("7. Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Test Accuracy: {accuracy:.4f}")
    print("   Classification report:")
    print(classification_report(y_test, y_pred))
    
    # 8. Save final model
    joblib.dump(model, PATHS['model'])
    print(f"✓ Model saved: {PATHS['model']}")
    
    # 9. Save configuration
    config = {
        'accuracy': float(accuracy),
        'training_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset_size': len(data),
        'features': {
            'tfidf_features': X_tfidf.shape[1],
            'svd_dimensions': svd.n_components,
            'total_features': X_combined.shape[1]
        },
        'classes': int(y.nunique()),
        'class_distribution': y.value_counts().to_dict()
    }
    
    import json
    with open(PATHS['config'], 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Configuration saved: {PATHS['config']}")
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return model, tfidf, svd

# =========================
# LOAD SAVED MODELS (For Prediction)
# =========================
def load_saved_models():
    """Load pre-trained models for prediction"""
    print("\n" + "="*60)
    print("LOADING SAVED MODELS")
    print("="*60)
    
    # Check if models exist
    for path_name, path in PATHS.items():
        if path_name != 'data' and not os.path.exists(path):
            print(f"✗ Missing: {path}")
            return None, None, None
    
    try:
        # Load models
        tfidf = joblib.load(PATHS['tfidf'])
        print(f"✓ TF-IDF loaded: {len(tfidf.vocabulary_)} features")
        
        svd = joblib.load(PATHS['svd'])
        print(f"✓ SVD loaded: {svd.n_components} components")
        
        model = joblib.load(PATHS['model'])
        print(f"✓ Classifier loaded")
        
        # Load config
        import json
        with open(PATHS['config'], 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✓ Configuration loaded (Accuracy: {config['accuracy']:.4f})")
        
        print("\n" + "="*60)
        print("MODELS LOADED SUCCESSFULLY!")
        print("="*60)
        
        return model, tfidf, svd
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return None, None, None

# =========================
# PREDICTION PIPELINE
# =========================
class HateSpeechDetector:
    """Production-ready hate speech detector (no gensim)"""
    
    LABEL_MAP = {
        0: "Non-Hate Speech | ሽክትንዙ ዊግትንድ አየውም",
        1: "Race Hate Speech | ዚረዝ ሽክትንዙ ዊግትንድ",
        2: "Religion Hate Speech | ሀይማኖትዝ ሽክትንዙ ዊግትንድ",
        3: "Gender Hate Speech | ጾተዝ ሽክትንዙ ዊግትንድ",
        4: "Political Hate Speech | ፖለቲከዝ ሽክትንዙ ዊግትንድ"
    }
    
    def __init__(self):
        """Initialize detector with saved models"""
        self.model, self.tfidf, self.svd = load_saved_models()
        self.is_loaded = all([self.model, self.tfidf, self.svd])
        
    def predict(self, text):
        """Predict hate speech category"""
        if not self.is_loaded:
            return "Error: Models not loaded"
        
        try:
            # 1. TF-IDF transformation
            tfidf_features = self.tfidf.transform([text])  # sparse (1, n_tfidf)
            
            # 2. SVD embedding from TF-IDF (dense)
            svd_embedding = self.svd.transform(tfidf_features)  # (1, n_components)
            svd_features = csr_matrix(svd_embedding)
            
            # 3. Combine features
            combined = hstack([tfidf_features, svd_features])
            
            # 4. Predict
            prediction = self.model.predict(combined)[0]
            
            # 5. Return readable label
            return self.LABEL_MAP.get(prediction, f"Unknown label: {prediction}")
            
        except Exception as e:
            return f"Prediction error: {str(e)}"
    
    def batch_predict(self, texts):
        """Predict multiple texts efficiently"""
        results = []
        for text in texts:
            results.append({
                'text': text,
                'prediction': self.predict(text)
            })
        return results

# =========================
# USAGE EXAMPLES
# =========================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("HATE SPEECH DETECTION SYSTEM")
    print("="*60)
    
    # OPTION 1: Train models (do this once)
    if not os.path.exists(PATHS['model']):
        print("No saved models found. Training new models...")
        model, tfidf, svd = train_and_save_models(svd_components=100)
    else:
        print("Saved models found. Loading...")
        model, tfidf, svd = load_saved_models()
    
    # OPTION 2: Use detector
    detector = HateSpeechDetector()
    
    if detector.is_loaded:
        # Test predictions
        test_texts = [
            "ሚዝረድ ቢርብጠስ",
            "ፅሊዝመ ቆርቆረዝ ይሽተኩ",
            "እስለመ ጅጥ አቕስተ ይሽተኩ",
            "እውነት አኹረመ ጉድ አኹረው ይሽተኩ",
            "ስልጣንል ፅበቁ እቅድ ዛሸነው"
        ]
        
        print("\n" + "="*60)
        print("TEST PREDICTIONS")
        print("="*60)
        
        for text in test_texts:
            result = detector.predict(text)
            print(f"\n📝 Text: {text}")
            print(f"✅ Prediction: {result}")
    
    print("\n" + "="*60)
    print("SYSTEM READY FOR DEPLOYMENT")
    print("="*60)