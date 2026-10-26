import re
import string
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Ensure required NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def initialize_environment():
    """Validates core packages are successfully imported."""
    print("[*] Environment Initialization Check:")
    import torch, transformers, sklearn
    print(f"    - NumPy version: {np.__version__}")
    print(f"    - Pandas version: {pd.__version__}")
    print(f"    - Scikit-Learn version: {sklearn.__version__}")
    print(f"    - PyTorch version: {torch.__version__}")
    print(f"    - Transformers version: {transformers.__version__}\n")

def load_sample_corpus():
    """
    Simulates Corpus Acquisition using mock security telemetry and logs.
    Replace this DataFrame with your loaded CSV/JSON security dataset (e.g., CICIDS or Syslogs).
    """
    print("[*] Corpus Acquisition & Mock Data Loading...")
    data = {
        "log_text": [
            "CRITICAL: Unauthorized exfiltration attempt detected via outbound SSH tunnel on port 22.",
            "WARNING: Potential privilege escalation observed. User 'admin_guest' executed sudo commands.",
            "INFO: Normal heartbeat connection established successfully from internal gateway.",
            "ERROR: Denial of Service (DoS) attack signature matched. High packet rate flood on port 80.",
            "CRITICAL: Sensitive data dump initiated from database server to external IP."
        ],
        "label": ["Exfiltration", "Privilege Escalation", "Benign", "Denial of Service", "Exfiltration"]
    }
    df = pd.DataFrame(data)
    print(f"    - Loaded {len(df)} sample log records successfully.\n")
    return df

def perform_eda(df):
    """Analyzes sequence lengths, missing features, and class skewness."""
    print("[*] Exploratory Data Analysis (EDA):")
    
    # Missing values check
    missing_counts = df.isnull().sum().sum()
    print(f"    - Missing values count: {missing_counts}")
    
    # Sequence length analysis (word count per log)
    df['word_count'] = df['log_text'].apply(lambda x: len(x.split()))
    print(f"    - Average sequence length: {df['word_count'].mean():.2f} words")
    
    # Class distribution / skewness
    class_counts = df['label'].value_counts()
    print("    - Class distribution:")
    for label, count in class_counts.items():
        print(f"        * {label}: {count} instance(s)")
    print()

def preprocess_text(text):
    """
    Text Preprocessing Pipeline:
    - Lowercasing
    - Punctuation stripping
    - Stop-word filtering
    - Lemmatization
    """
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Punctuation stripping
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Tokenization & Stop-word filtering + Lemmatizer
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    tokens = text.split()
    cleaned_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words
    ]
    
    return " ".join(cleaned_tokens)

def run_feature_extraction(df):
    """Applies TF-IDF vectorizer to the preprocessed logs."""
    print("[*] Text Preprocessing & Feature Extraction Pipeline:")
    
    # Apply cleaning function to corpus
    df['cleaned_text'] = df['log_text'].apply(preprocess_text)
    
    for orig, cleaned in zip(df['log_text'], df['cleaned_text']):
        print(f"    RAW:     {orig}")
        print(f"    CLEANED: {cleaned}\n")
        
    # Feature Extraction via TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
    
    print(f"    - TF-IDF Matrix generated with shape: {tfidf_matrix.shape}")
    print(f"    - Vocabulary size: {len(vectorizer.get_feature_names_out())}\n")
    return tfidf_matrix, vectorizer

def train_classifier(df, tfidf_matrix):
    """Splits data, trains a classifier, and evaluates performance."""
    print("[*] Model Training & Evaluation Pipeline:")
    
    X = tfidf_matrix
    y = df['label']
    
    # Train-test split (with check for small class counts)
    test_size = 0.2 if len(df) >= 5 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Using class_weight='balanced' to handle potential skewed data
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("    - Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return model

def save_artifacts(model, vectorizer, model_path="model.pkl", vec_path="vectorizer.pkl"):
    """Saves the trained model and vectorizer for inference."""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"[*] Artifacts saved successfully to {model_path} and {vec_path}")

def predict_threat(new_log, model, vectorizer):
    """Classifies a new incoming log message with confidence scoring."""
    cleaned = preprocess_text(new_log)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)
    confidence = np.max(probabilities)
    
    return prediction, confidence

if __name__ == "__main__":
    initialize_environment()
    df_corpus = load_sample_corpus()
    perform_eda(df_corpus)
    
    # Run feature extraction and get both matrix and vectorizer
    tfidf_matrix, vectorizer = run_feature_extraction(df_corpus)
    
    # Train model
    trained_model = train_classifier(df_corpus, tfidf_matrix)
    
    # Save artifacts for production/inference use
    save_artifacts(trained_model, vectorizer)
    
    # Test real-time inference
    test_log = "CRITICAL: External IP attempting rapid connection flood on port 443."
    pred_label, conf = predict_threat(test_log, trained_model, vectorizer)
    print(f"[*] Test Inference:")
    print(f"    - Log: {test_log}")
    print(f"    - Prediction: {pred_label} (Confidence: {conf:.2f})")
