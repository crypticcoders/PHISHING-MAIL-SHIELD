# 🛡️ AI-Driven Phishing Email Detection Using NLP

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Made with Colab](https://img.shields.io/badge/Made%20with-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Status](https://img.shields.io/badge/Status-Complete-success)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#)

> An end-to-end machine learning pipeline that detects phishing emails by combining **TF-IDF text features** with **engineered structural signals**, evaluated across four classifiers and fused into a majority-vote ensemble.

Built as **Project-2** of the AI & ML Summer Internship Program at the **Indian Institute of Computing and Technology (IICT)** — 15 June to 30 July 2026.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Methodology](#-methodology)
  - [1. Text Preprocessing](#1-text-preprocessing)
  - [2. Structural Metadata Features](#2-structural-metadata-features)
  - [3. Feature Fusion](#3-feature-fusion)
  - [4. Model Training](#4-model-training)
  - [5. Ensemble Voting](#5-ensemble-voting)
- [Results & Visualizations](#-results--visualizations)
- [Comparative Model Analysis](#-comparative-model-analysis)
- [Live Ensemble Testing](#-live-ensemble-testing)
- [Feature Importance](#-feature-importance)
- [Challenges Faced](#-challenges-faced)
- [What I Learned](#-what-i-learned)
- [Future Scope](#-future-scope)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [References](#-references)
- [Author](#-author)

---

## 🔍 Overview

Phishing exploits human trust, not software vulnerabilities — which is exactly why static, rule-based spam filters (blacklists, keyword matching) keep losing to increasingly well-written, LLM-assisted phishing campaigns. This project builds a **supervised ML system** that reads an email the way a human would: it looks at *what the email says* (TF-IDF over cleaned body text) and *how it's constructed* (presence of URLs, urgency-triggering language).

**What the pipeline does, end to end:**
1. Ingests and label-normalizes a 15,000-email corpus (phishing vs. legitimate).
2. Cleans text (HTML stripping, URL masking, stop-word removal) and engineers two structural metadata features.
3. Vectorizes text with TF-IDF (2,500 terms) and fuses it with metadata → a 2,502-dimensional feature space.
4. Trains and rigorously compares **four classifiers** — Logistic Regression, Random Forest, Gaussian Naive Bayes, and a Neural Network (MLP) — on identical train/test splits.
5. Combines the three strongest models into a **majority-vote ensemble** for more robust real-world inference.
6. Validates everything on unseen, hand-written sample emails — including a genuinely ambiguous case.

The goal wasn't just a high accuracy number — it was building a real, reproducible understanding of every stage of an applied NLP system: the trade-off between interpretability (Logistic Regression), non-linear ensemble strength (Random Forest), probabilistic simplicity (Naive Bayes), and representational flexibility (Neural Networks).

---

## 🏆 Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** 🥇 | **96.77%** | **0.97** | **0.97** | **0.97** |
| **Neural Network (MLP)** | 96.50% | 0.97 | 0.96 | 0.96 |
| **Logistic Regression** | 96.00% | 0.96 | 0.96 | 0.96 |
| **Gaussian Naive Bayes** | 91.90% | 0.92 | 0.92 | 0.92 |

- 📊 Evaluated on a **3,000-email held-out test set** (1,462 legitimate / 1,538 phishing)
- 🎯 **Random Forest** — best overall accuracy, lowest false-positive count (44), highest AUC (0.9891)
- ⚠️ **Naive Bayes** — highest false-negative rate (199 missed phishing emails, ~13%), excluded from the final ensemble on this basis
- 🗳️ **Ensemble (LR + RF + NN, majority vote)** — correctly resolved a genuinely ambiguous unseen sample through model disagreement

---

## 🧰 Tech Stack

| Library | Purpose |
| :--- | :--- |
| `pandas` | Data loading, cleaning, and manipulation |
| `scikit-learn 1.9.0` | TF-IDF vectorization, train/test split, all 4 classifiers, evaluation metrics |
| `scipy.sparse` | Efficient fusion of the sparse TF-IDF matrix with dense metadata columns |
| `NumPy` | Array operations, feature-importance ranking |
| `Matplotlib` / `Seaborn` | Confusion matrices, ROC curves, performance charts |
| `re` (Regex) | HTML stripping, URL detection/masking, punctuation removal |
| `pickle` | Persisting trained models + fitted vectorizer for reuse |

**Environment:** Google Colab (zero-config runtime, native Google Drive integration)

---

## 📂 Dataset

<p align="center">
  <img src="assets/01_class_distribution.png" alt="Class distribution of the working dataset" width="480">
</p>

| Attribute | Value |
| :--- | :--- |
| **Total emails (working subset)** | 15,000 |
| **Phishing emails (label = 1)** | 7,795 (51.97%) |
| **Legitimate emails (label = 0)** | 7,205 (48.03%) |
| **Training set (80%)** | 12,000 emails |
| **Test set (20%)** | 3,000 emails (Safe: 1,462 · Phishing: 1,538) |
| **Random seed** | 42 (fixed for reproducibility) |

**Label normalization:** Raw string variants (`"spam"`, `"phishing"`, `"1"`, `"1.0"` → `1`; `"ham"`, `"legitimate"`, `"0"`, `"0.0"` → `0`) were programmatically mapped, and rows with unmappable labels or empty bodies were dropped.

**A genuinely hard negative class:** The legitimate samples were dominated by terse, jargon-heavy operational correspondence (logistics/scheduling notices) — not friendly small talk — forcing the model to learn real discriminative signal rather than simply flagging "external-sounding" language.

---

## 🏗️ Pipeline Architecture

<p align="center">
  <img src="assets/02_architecture_pipeline.png" alt="End-to-end architectural pipeline" width="800">
</p>

```text
Raw Email Input
      │
      ▼
Text Preprocessing (RegEx / Stopwords)
      │
      ▼
Feature Extraction (TF-IDF + Metadata)
      │
      ▼
Model Training (LR, RF, NB, NN)
      │
      ▼
Ensemble Vote → PHISHING DETECTED / SAFE-LEGITIMATE
```

### 1. Text Preprocessing

Email bodies undergo a strict cleaning order to remove noise while preserving semantic intent:

```python
def preprocess_email_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)                          # Strip HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', ' url_token ', text)   # Normalize URLs
    text = re.sub(r'[^a-z\s]', '', text)                          # Strip numbers/punctuation
    words = text.split()
```

### 2. Structural Metadata Features

Computed on the **raw (uncleaned)** body to capture explicit structural and psychological cues:
- `has_url` — Binary flag (`1` if the body contains an `http(s)://` or `www.` URL, else `0`).
- `urgency_words` — Binary flag (`1` if the body contains urgency/authority triggers: *urgent, verify, suspend, action, password, login, bank, account*).

---

### 3. Feature Fusion

Cleaned text is vectorized using TF-IDF (top 2,500 terms) and combined with dense structural metadata into a **2,502-dimensional feature space**:

```python
vectorizer = TfidfVectorizer(max_features=2500)
X_tfidf = vectorizer.fit_transform(df_subset['clean_body'])
metadata_features = csr_matrix(df_subset[['has_url', 'urgency_words']].values)
X_combined = hstack([X_tfidf, metadata_features]).toarray()  # 2,502 dimensions
```

    return ' '.join([w for w in words if w not in STOPWORDS])

4. Model Training
Four algorithmically distinct classifiers were trained on the identical 12,000-row training matrix to ensure fair evaluation:
```
models = {
    "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=30, random_state=42)
}
```
5. Ensemble Voting
The three top-performing classifiers (Logistic Regression, Random Forest, and Neural Network) cast a binary vote during inference; the majority decision determines the final label. Naive Bayes was deliberately excluded due to its high false-negative rate.

---

## 📊 Results & Visualizations

### Performance Comparison

<p align="center">
  <img src="assets/03_performance_comparison.png" alt="Performance Comparison" width="700">
</p>

### Confusion Matrices

<p align="center">
  <img src="assets/04_confusion_matrices.png" alt="Confusion Matrices" width="700">
</p>

| Model | True Negative | False Positive | False Negative | True Positive |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 1,404 | 58 | 62 | 1,476 |
| **Random Forest** 🥇 | **1,418** | **44** | 53 | 1,485 |
| **Naive Bayes** | 1,418 | 44 | **199** ⚠️ | 1,339 |
| **Neural Network** | 1,404 | 58 | **47** | **1,491** |

> **Security Context:** A **false negative** (missing a phishing email) is exponentially more dangerous than a false positive. Naive Bayes missed nearly 4× more phishing emails than any other model, driving the decision to exclude it from the final ensemble.

---

### ROC / AUC Analysis

<p align="center">
  <img src="assets/05_roc_curves.png" alt="ROC Curves" width="600">
</p>

| Model | AUC |
| :--- | :---: |
| **Random Forest** | **0.9891** |
| **Neural Network** | 0.9862 |
| **Logistic Regression** | 0.9838 |
| **Naive Bayes** | 0.9540 |

---

## ⚖️ Comparative Model Analysis

| Rank | Model | Key Strength | Key Weakness |
| :---: | :--- | :--- | :--- |
| 🥇 **1** | **Random Forest** | Highest overall accuracy, lowest false positives, interpretable via Gini importance | Larger serialized file size |
| 🥈 **2** | **Neural Network** | Highest phishing recall (lowest false negatives), strong non-linear feature capture | Black-box behavior, higher compute overhead |
| 🥉 **3** | **Logistic Regression** | High accuracy, fast retraining time, fully interpretable via coefficients | Linear boundary constraint |
| **4** | **Naive Bayes** | Instant training time, lightweight probabilistic baseline | Unrealistic feature-independence assumption breaks under correlated TF-IDF terms |

---

## 🧪 Live Ensemble Testing

Four unseen, hand-written test emails were evaluated through the ensemble pipeline:

| Sample Email Context | LR | RF | NN | Ensemble Decision |
| :--- | :---: | :---: | :---: | :---: |
| *"Quarterly engineering sync rescheduled to Thursday..."* | Safe | Safe | Safe | ✅ **SAFE / LEGITIMATE** |
| *"FINAL NOTICE: tax filing discrepancies, update at http://refund-portal-gov.net..."* | Phish | Phish | Phish | 🚨 **PHISHING DETECTED** |
| *"Your package has shipped via standard transit, track on carrier portal..."* | Safe | **Phish** | Safe | ✅ **SAFE / LEGITIMATE** |
| *"SECURITY ALERT: unauthorised login to your banking app, reset at http://login-auth-verification.com..."* | Phish | Phish | Phish | 🚨 **PHISHING DETECTED** |

> **Key Takeaway:** In the third sample, Random Forest over-indexed on the tracking link and misclassified the email as phishing. LR and NN correctly identified it as benign, allowing the majority vote to prevent a false alarm.

---

## 🔬 Feature Importance

<p align="center">
  <img src="assets/06_feature_importance.png" alt="Top Feature Importance" width="700">
</p>

The engineered metadata features (`urgency_words`, `has_url`) ranked among the **most influential predictors overall**, confirming that structural cues carry critical signal alongside lexical terms (`verify`, `account`, `suspend`, `password`, `login`, `bank`).

---

## 🚧 Challenges Faced

- **Class Boundary Ambiguity:** Operational emails (shipping alerts, system logs) frequently use compressed, urgent tone similar to phishing tactics.
- **Matrix Tractability:** Balancing TF-IDF vocabulary size (2,500 max features) to ensure memory efficiency during dense matrix operations.
- **Algorithmic Bias in Naive Bayes:** Analyzing how feature-independence assumptions collapse when handling sparse, high-dimensional TF-IDF matrices.

---

## 🎓 What I Learned

- Designing end-to-end NLP pipelines for cybersecurity threat classification.
- Balancing model interpretability against predictive power when selecting models for an ensemble.
- Evaluating security classifiers based on domain-specific risks (prioritizing Recall to minimize False Negatives).

---

## 🔭 Future Scope

- [ ] Deploy the pipeline as a live web application using **Streamlit**.
- [ ] Upgrade feature extraction from TF-IDF to transformer embeddings (**BERT** / **RoBERTa**).
- [ ] Integrate sender reputation metrics (SPF / DKIM / DMARC verification).

---

## 📁 Project Structure

```text
phishing-email-detection/
├── assets/
│   ├── 01_class_distribution.png
│   ├── 02_architecture_pipeline.png
│   ├── 03_performance_comparison.png
│   ├── 04_confusion_matrices.png
│   ├── 05_roc_curves.png
│   └── 06_feature_importance.png
├── notebook/
│   └── phishing_email_detection.ipynb
├── models/
│   ├── phishing_models.pkl
│   └── phishing_vectorizer.pkl
├── report/
│   └── PHISHING_MAIL_REPORT.pdf
├── README.md
└── requirements.txt
