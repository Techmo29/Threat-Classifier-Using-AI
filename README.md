# Threat-Classifier-Using-AI

A machine learning and natural language processing pipeline for automated cybersecurity threat categorization, log analysis, and incident triage.

---

## One-Month Technical Roadmap

### Week 1: Data Ingestion, Exploration & Preprocessing
*Objective: Establish the data pipeline, parse structured/unstructured telemetry, and handle class imbalance.*

- [ ] **Environment Initialization:** Configure Python virtual environment, install core numerical and machine learning packages (`numpy`, `pandas`, `scikit-learn`, `nltk`, `torch`, `transformers`).
- [ ] **Corpus Acquisition:** Source standardized security datasets (e.g., CICIDS intrusion datasets, MITRE ATT&CK technique descriptions, or parsed Syslog/Windows Event log repositories).
- [ ] **Exploratory Data Analysis (EDA):** Analyze token frequency distributions, sequence lengths, missing features, and class skewness across threat categories (e.g., Exfiltration, Privilege Escalation, Denial of Service).
- [ ] **Text Preprocessing Pipeline:** Implement lowercasing, punctuation stripping, stop-word filtering, lemmatization, and feature extraction via TF-IDF vectorization or subword tokenization (Byte-Pair Encoding).

### Week 2: Baseline Modeling & Transformer Fine-Tuning
*Objective: Establish statistical baselines and benchmark deep learning architectures.*

- [ ] **Statistical Baselines:** Train traditional supervised classifiers (Logistic Regression, Linear Support Vector Machines, Random Forest) using sparse TF-IDF matrices. Evaluate baseline accuracy, precision, recall, and macro F1-scores using stratified K-fold cross-validation.
- [ ] **Neural Architecture / Fine-Tuning:** Implement a sequence classification pipeline using a pre-trained transformer encoder (e.g., `distilbert-base-uncased` or a security-focused model like `SecBERT`). Fine-tune the model heads with cross-entropy loss.
- [ ] **Threshold Optimization:** Tune classification confidence thresholds to minimize false negatives on critical-severity classes.

### Week 3: Model Serialization & REST API Integration
*Objective: Package the inference pipeline into a high-throughput backend service.*

- [ ] **Artifact Persistence:** Serialize model weights, configuration parameters, and the vectorizer/tokenizer pipeline using `joblib` or PyTorch state dictionaries.
- [ ] **Inference Wrapper:** Write a modular Python wrapper class that handles batch preprocessing and inference execution.
- [ ] **API Development:** Build a asynchronous REST API using **FastAPI** with input validation via Pydantic models. Implement endpoints for single log payload scoring (`/predict`) and batch processing (`/batch-predict`).

### Week 4: Containerization, Testing & CI/CD
*Objective: Secure, containerize, and test the microservice.*

- [ ] **Unit & Integration Testing:** Write automated test suites using `pytest` to validate preprocessing outputs, model dimension matching, and HTTP status codes.
- [ ] **Containerization:** Write a multi-stage `Dockerfile` optimizing image size for Python deep learning dependencies.
- [ ] **Logging & Monitoring:** Implement structured JSON logging within the API to trace request execution times and input validation errors.

---
## References
1. https://eprints.hud.ac.uk/id/eprint/24330/6/MohammadPhishing14July2015.pdf

## Getting Started

### Prerequisites
* Python 3.9+
* Pip and virtualenv

### Installation

1. **Clone the repository:**
   ```bash
   git clone [Threat-Classifier-Using-AI](https://github.com/Techmo29/Threat-Classifier-Using-AI.git)
   cd Threat-Classifier-Using-AI
