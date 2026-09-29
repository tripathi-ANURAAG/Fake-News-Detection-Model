# Fake-News-Detection-Model
# Fake News Classification & Inference Engine

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/scikit--learn-NLP-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning solution designed to classify news headlines and articles as real or fake. Built with custom modular object-oriented components for preprocessing, evaluation, and inference pipeline execution.

---

## 📌 Key Impact Metrics

* **Dataset Scale:** Processed **72,134 raw records** down to **71,537 cleaned, non-null text records**.
* **Vocabulary Matrix:** Extracted **5,000 TF-IDF features** (`float32` precision) across the entire corpus.
* **Generalization Performance:** Achieved **99.98% Training Accuracy** and **89.87% Testing Accuracy** on unseen test instances.
* **Dataset Partitioning:** Stratified **77/23 split** preserving class balances (55,083 train samples / 16,454 test samples).
* **Production Readability:** Modularized runtime logic via dedicated `Evaluation` and `Preprocessing` pipeline classes for real-time inference handling.

---

## 🏗️ End-to-End System Architecture
+-----------------------------------------------------------------------------------+
|                                 TRAINING PIPELINE                                 |
+--------------------+      +--------------------+      +---------------------------+
| WELFake Dataset    | ---> | Regex & Stopwords  | ---> | TF-IDF Vectorizer         |
| (72,134 Raw Rows)  |      | Cleaning (NLTK)    |      | (5,000 Max Features)      |
+--------------------+      +--------------------+      +---------------------------+
                                                                      |
                                                                      v
+--------------------+      +--------------------+      +---------------------------+
| Evaluation Metrics | <--- | Random Forest      | <--- | Stratified Train/Test     |
| (89.87% Test Acc)  |      | Classifier Model   |      | Split (77/23 Ratio)       |
+--------------------+      +--------------------+      +---------------------------+
                                                                      |
+---------------------------------------------------------------------v-------------+
|                                INFERENCE PIPELINE                                 |
+-----------------------------------------------------------------------------------+
| Raw String Input ---> Preprocessing Class (NLTK) ---> Model Predict ---> Output   |
| "Police turned to..."                                     ("THE NEWS IS REAL")   |
+-----------------------------------------------------------------------------------+

---

## 📂 Project Structure
├── data/
│   └── WELFake_Dataset.csv         # Dataset source containing headlines and news text
├── notebooks/
│   └── FakeNewsClassification.ipynb # Complete Notebook implementation
├── README.md                       # Documentation & Project Overview
└── requirements.txt                # Dependencies

---

## ⚙️ Technical Methodology & Pipeline Modules

### 1. Data Ingestion & Cleaning
* Ingested `WELFake_Dataset.csv` containing `72,134` entries across `4` primary columns: `Unnamed: 0`, `title`, `text`, and `label`.
* Identified and removed missing records (`558` missing titles, `39` missing text instances), reducing the working set to `71,537` validated records.
* Class distribution maintained balance with `37,106` positive (`1`) labels and `35,028` negative (`0`) labels.

### 2. Modular NLP Preprocessing (`Preprocessing` Class)
Implemented an OOP encapsulation class for real-time text transformation:
* **Pattern Extraction:** Uses regular expressions (`re.sub('[^a-zA-Z]', ' ', text)`) to purge numerical and special characters.
* **Lemmatization:** Utilizes `WordNetLemmatizer` for token normalizing combined with NLTK English stop-word filtering.

PYTHON
class Preprocessing:
    def __init__(self, data):
        self.data = data
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords_list = stopwords.words('english')
        
    def preprocess_data(self):
        # Cleans, tokenizes, filters stop-words, and lemmatizes input strings
        ...


### 3. Feature Vectorization & Train Split
Feature matrix generated using TfidfVectorizer(max_features=5000, dtype=np.float32).

Resulting matrix shape: (71537, 5000).

Split strategy: train_test_split(X, y, test_size=0.23, random_state=10, stratify=y).


### 4. Model Training & Validation (Evaluation Class)
Model architecture built on RandomForestClassifier with multi-threading enabled (n_jobs=-1).

| Dataset Split | Precision (Class 0) | Recall (Class 0) | Precision (Class 1) | Recall (Class 1) | Overall Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 1.00 | 1.00 | 1.00 | 1.00 | **99.98%** |
| **Test Set** | 0.90 | 0.89 | 0.89 | 0.91 | **89.87%** |


🔮 Production Inference Class (Prediction Class)
The system features a dedicated inference pipeline wrapper that accepts raw text strings, executes runtime feature cleaning, and outputs clear classifications.
class Prediction:
    def __init__(self, prod_data):
        self.prod_data = prod_data
        self.model = model
        
    def prediction_news(self):
        preprocessed_data = Preprocessing(self.prod_data).preprocess_data()
        X_test = tf.transform(preprocessed_data).toarray()
        prediction = self.model.predict(X_test)
        
        if prediction[0] == 0:
            return "THE NEWS IS FAKE"
        else:
            return "THE NEWS IS REAL"

Inference Sample Run
* Input Text: "Police turn to Badger Power for India Welfare Protest Standing Park Violence"   
* Output: "THE NEWS IS REAL"

🚀 Environment Setup & Execution

Prerequisites
* Python 3.8+
* Jupyter Notebook / JupyterLab

🛠️ Technology Stack
* Programming Language: Python 3.8+

* Data Processing: Pandas, NumPy

* Text Processing & NLP: NLTK (WordNetLemmatizer, stopwords)

* Machine Learning Pipeline: Scikit-Learn (TfidfVectorizer, RandomForestClassifier)
