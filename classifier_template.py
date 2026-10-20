from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Sample training data: threat logs/descriptions and their labels
X_train = [
    'Failed login attempt from unauthorized IP address',
    'Multiple root access failures detected',
    'Outbound data exfiltration to unknown external server',
    'Normal HTTP GET request to homepage',
    'User successfully logged in via single sign-on',
    'Abnormal spike in outbound traffic volume',
]

y_train = [
    'Brute Force',
    'Brute Force',
    'Exfiltration',
    'Benign',
    'Benign',
    'Exfiltration',
]

# 2. Build a pipeline combining TF-IDF feature extraction and Logistic Regression
model = Pipeline(
    [
        ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
        ('clf', LogisticRegression()),
    ]
)

# 3. Train the model
model.fit(X_train, y_train)

# 4. Test the model with new log samples
new_logs = [
    'Repeated password failures from remote host',
    'Standard CSS style sheet requested',
]

predictions = model.predict(new_logs)

for log, prediction in zip(new_logs, predictions):
  print(f'Log: "{log}" ---> Predicted Threat: {prediction}')
