import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, make_scorer, f1_score

class ThreatBaselineTrainer:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        # Define baseline models to benchmark
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "Linear SVM": LinearSVC(class_weight='balanced', random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        }

    def evaluate_baselines(self, texts: list, labels: list):
        """Vectorizes text and evaluates baseline classifiers using Stratified K-Fold."""
        print("[*] Generating TF-IDF feature matrix...")
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n[*] Training and evaluating: {name}...")
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
            
            results[name] = {
                "Accuracy": np.mean(cv_results['test_accuracy']),
                "Macro Precision": np.mean(cv_results['test_precision_macro']),
                "Macro Recall": np.mean(cv_results['test_recall_macro']),
                "Macro F1": np.mean(cv_results['test_f1_macro'])
            }
            
        return results

# Example execution workflow
if __name__ == "__main__":
    # Sample corpus mimicking categorized threat logs
    sample_data = {
        "text": [
            "unauthorized ssh login attempt detected from external ip",
            "sudoers file modified by low privileged account privilege escalation",
            "massive outbound https traffic transfer to unknown external server exfiltration",
            "repeated failed login attempts brute force attack in progress",
            "root shell spawned via vulnerable web application exploit"
        ],
        "label": ["Brute Force", "Privilege Escalation", "Exfiltration", "Brute Force", "Privilege Escalation"]
    }
    
    df = pd.DataFrame(sample_data)
    
    trainer = ThreatBaselineTrainer(n_splits=2) # Using 2 splits just for tiny sample demonstration
    performance_metrics = trainer.evaluate_baselines(df['text'], df['label'])
    
    print("\n====================")
    print("BASELINE MODEL SUMMARY")
    print("====================")
    metrics_df = pd.DataFrame(performance_metrics).T
    print(metrics_df.to_string())
