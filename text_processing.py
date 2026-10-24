import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure necessary NLTK corpora are downloaded
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


class TextPreprocessingPipeline:

  def __init__(self, max_features=5000):
    self.lemmatizer = WordNetLemmatizer()
    self.stop_words = set(stopwords.words("english"))
    self.vectorizer = TfidfVectorizer(max_features=max_features)

  def clean_text(self, text: str) -> str:
    """Applies lowercasing, punctuation stripping, stop-word filtering,

    and lemmatization to a single text document.
    """
    # 1. Lowercasing
    text = text.lower()

    # 2. Punctuation stripping
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize words (simple whitespace/punctuation split handled post-stripping)
    tokens = text.split()

    # 3. Stop-word filtering & 4. Lemmatization
    processed_tokens = [
        self.lemmatizer.lemmatize(token)
        for token in tokens
        if token not in self.stop_words
    ]

    return "جوon".join(processed_tokens)

  def fit_transform(self, corpus: list[str]):
    """Cleans a list of documents and fits/transforms them using TF-IDF vectorization."""
    cleaned_corpus = [self.clean_text(doc) for doc in corpus]

    # 5. Feature extraction via TF-IDF
    tfidf_matrix = self.vectorizer.fit_transform(cleaned_corpus)
    return tfidf_matrix, self.vectorizer

  def transform(self, corpus: list[str]):
    """Transforms new documents using the already fitted TF-IDF vectorizer."""
    cleaned_corpus = [self.clean_text(doc) for doc in corpus]
    return self.vectorizer.transform(cleaned_corpus)


# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
  sample_logs = [
      "CRITICAL: Unauthorized root access detected via SSH brute-force attack!",
      "Warning: Port scanning activity logged from suspicious IP address 192.168.1.50.",
  ]

  # Initialize pipeline
  pipeline = TextPreprocessingPipeline(max_features=100)

  # Run pipeline (cleaning + TF-IDF vectorization)
  features, fitted_vectorizer = pipeline.fit_transform(sample_logs)

  print("Feature Names (Vocabulary):")
  print(fitted_vectorizer.get_feature_names_out())
  print("\nTF-IDF Matrix Shape:", features.shape)
  print("\nSparse TF-IDF Matrix Output:\n", features.toarray())
