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
⚙️ Methodology1. Text PreprocessingPythondef preprocess_email_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)                          # strip HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', ' url_token ', text)   # normalize URLs
    text = re.sub(r'[^a-z\s]', '', text)                          # strip numbers/punctuation
    words = text.split()
    return ' '.join([w for w in words if w not in STOPWORDS])
2. Structural Metadata FeaturesComputed on the raw (uncleaned) body, since these are structural, not lexical:has_url — 1 if the body contains an http(s):// or www. patternurgency_words — 1 if the body contains any of: urgent, verify, suspend, action, password, login, bank, account3. Feature FusionPythonvectorizer = TfidfVectorizer(max_features=2500)
X_tfidf = vectorizer.fit_transform(df_subset['clean_body'])
metadata_features = csr_matrix(df_subset[['has_url', 'urgency_words']].values)
X_combined = hstack([X_tfidf, metadata_features]).toarray()  # → 2,502 dimensions
4. Model TrainingPythonmodels = {
    "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=30, random_state=42)
}
All four models were trained on the identical 12,000-row training matrix and evaluated on the identical 3,000-row test matrix — so any performance gap is attributable purely to algorithmic inductive bias, not data inconsistency.5. Ensemble VotingThe three strongest models (LR, RF, NN) cast a binary vote each; the majority label wins. Naive Bayes was deliberately excluded due to its weaker standalone performance.📊 Results & VisualizationsPerformance ComparisonConfusion MatricesModelTrue NegativeFalse PositiveFalse NegativeTrue PositiveLogistic Regression1,40458621,476Random Forest1,41844531,485Naive Bayes1,41844199 ⚠️1,339Neural Network1,40458471,491In a security context, a false negative (missed phishing email) is far costlier than a false positive. Naive Bayes lets through ~4× more phishing emails than the other three models — the single biggest reason it was excluded from the ensemble.ROC / AUC AnalysisModelAUCRandom Forest0.9891Neural Network0.9862Logistic Regression0.9838Naive Bayes0.9540⚖️ Comparative Model AnalysisRankModelKey StrengthKey Weakness🥇 1Random ForestHighest accuracy, lowest false positives, interpretable via feature importanceLarger model size, slightly slower inference🥈 2Neural NetworkHighest phishing recall, strong non-linear fitNo built-in interpretability, sensitive to tuning🥉 3Logistic RegressionExcellent accuracy-to-simplicity ratio, fully interpretable, fastest to retrainPurely linear decision boundary4Naive BayesExtremely fast training, useful probabilistic baselineHigh false-negative rate, unrealistic feature-independence assumptionWhy Naive Bayes underperforms here: Its precision on the Safe class (0.88) is notably lower than on Phishing (0.97), while recall shows the opposite pattern — a textbook symptom of the conditional-independence assumption breaking down against thousands of correlated TF-IDF features.🧪 Live Ensemble TestingFour hand-written sample emails — never seen during training — were run through the full pipeline:Sample (truncated)LRRFNNEnsemble Decision"Quarterly engineering sync rescheduled to Thursday..."SafeSafeSafe✅ SAFE / LEGITIMATE"FINAL NOTICE: tax filing discrepancies, update at http://refund-portal-gov.net..."PhishPhishPhish🚨 PHISHING DETECTED"Your package has shipped via standard transit, track on carrier portal..."SafePhishSafe✅ SAFE / LEGITIMATE"SECURITY ALERT: unauthorised login to your banking app, reset at http://login-auth-verification.com..."PhishPhishPhish🚨 PHISHING DETECTEDThe shipping-notification sample is the most interesting case: Random Forest alone flagged it as phishing (likely over-weighting the presence of a tracking link), while LR and NN correctly read it as benign. This is exactly why the ensemble exists — it averages out any single model's occasional over-sensitivity to one structural cue instead of trusting it in isolation.🔬 Feature ImportanceThe engineered metadata features (urgency_words, has_url) rank among the most influential predictors overall — validating the decision to augment pure TF-IDF text with explicit structural signals. Among lexical terms, classic social-engineering pressure words dominate: verify, account, suspend, password, login, click, bank, confirm. The model independently rediscovered these patterns purely from labelled data, without ever being told which words were "suspicious."🚧 Challenges FacedAmbiguous legitimate samples — Terse, jargon-heavy operational emails occasionally resembled the compressed, action-oriented tone of phishing text, raising the real difficulty of the classification boundary.Dimensionality vs. training time — Capping the TF-IDF vocabulary at 2,500 terms kept the dense 12,000 × 2,502 combined matrix tractable in Colab, particularly for Naive Bayes and the MLP.Neural Network convergence — The MLP raised a ConvergenceWarning at the 30-iteration cap, a deliberate, documented training-time/accuracy trade-off rather than an oversight.Fair evaluation protocol — A fixed random_state=42 across every split was essential so that performance differences reflect the algorithms themselves, not accidental data-split variance.Environment consistency — scikit-learn was explicitly pinned/upgraded to 1.9.0 to avoid subtle API drift between Colab's default install and the development environment.Interpreting Naive Bayes' asymmetric errors — Tracing the lopsided precision/recall split back to the conditional-independence assumption was a genuinely useful exercise in why classical model assumptions matter in practice, not just in theory.🎓 What I LearnedNLP for cybersecurity — How raw language gets turned into a structured, quantitative signal, and how phishing's reliance on urgency and authority-mimicry leaves a statistically detectable fingerprint.Rigorous model evaluation — Training four algorithmically distinct classifiers on identical data made the causal link between model architecture and measurable outcomes (accuracy, false-negative rate) concrete, not theoretical.Ethical AI in threat detection — The asymmetric cost of errors in security (missed phishing >> false alarm), the risk of a filter penalizing certain writing styles or non-native phrasing, and the importance of a human-in-the-loop before auto-quarantining mail.Technical communication — Structuring findings into a formal, IEEE-style report end to end: abstract, literature review, numbered tables/figures, and a properly cited reference list.🔭 Future Scope[ ] Deploy the ensemble behind a lightweight Streamlit web interface for instant, paste-and-classify inference.[ ] Replace/augment TF-IDF with contextual embeddings (e.g., BERT) to catch semantic phishing cues that survive paraphrasing.[ ] Extend MLP training budget (max_iter beyond 30) and run systematic hyperparameter tuning across all four models.[ ] Add sender-domain reputation and SPF/DKIM/DMARC header metadata as additional structural features.[ ] Stress-test against adversarial phishing emails crafted to evade high-importance keywords, then retrain on adversarial examples.[ ] Move to a larger, continuously updated corpus to guard against phishing-language concept drift over time.📁 Project StructurePlaintextphishing-email-detection/
├── assets/
│   ├── 01_class_distribution.png
│   ├── 02_architecture_pipeline.png
│   ├── 03_performance_comparison.png
│   ├── 04_confusion_matrices.png
│   ├── 05_roc_curves.png
│   └── 06_feature_importance.png
├── notebook/
│   └── phishing_email_detection.ipynb     # Full Colab notebook
├── models/
│   ├── phishing_models.pkl                # Pickled dict of 4 trained models
│   └── phishing_vectorizer.pkl            # Fitted TF-IDF vectorizer
├── report/
│   └── PHISHING_MAIL_REPORT.pdf           # Full IEEE-style project report
├── README.md
└── requirements.txt
🚀 Getting StartedPrerequisitesBashpip install --upgrade scikit-learn==1.9.0 pandas numpy scipy matplotlib seaborn
Run the PipelinePython# 1. Load & normalize labels
df = pd.read_csv('phishing_email.csv')

# 2. Preprocess + engineer metadata features
df['clean_body'] = df['body'].apply(preprocess_email_text)

# 3. Vectorize + fuse features, train/test split
# 4. Train all four models
# 5. Run live inference
result, votes = test_phishing_email_live("Your email text here...")
print(result)  # 1 = phishing, 0 = safe
Full, unabridged, cell-by-cell source code is available in report/PHISHING_MAIL_REPORT.pdf and the accompanying Colab notebook.Quick Inference ExamplePythontest_phishing_email_live(
    "SECURITY ALERT: unauthorised login to your banking app detected. "
    "Reset your password immediately at [http://login-auth-verification.com](http://login-auth-verification.com)"
)
# → Model Votes [LR, RF, NN]: [1, 1, 1]
# → ENSEMBLE DECISION: PHISHING EMAIL DETECTED
📚 ReferencesF. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, vol. 12, pp. 2825–2830, 2011.L. Breiman, "Random Forests," Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.I. Fette, N. Sadeh, and A. Tomasic, "Learning to detect phishing emails," in Proc. WWW, 2007, pp. 649–656.A. Y. Ng, "On discriminative vs. generative classifiers: A comparison of logistic regression and naive Bayes," NeurIPS, 2001.Anti-Phishing Working Group (APWG), "Phishing Activity Trends Report," 2025.Kaggle, "Phishing Email Dataset," Kaggle Datasets Repository, 2024.👤 AuthorChristy Joyce AECE, VIT Chennai · AI & ML Summer Intern, IICT
---

<ElicitationsGroup message="What would you like to do next?">
<Elicitation label="Draft a Streamlit app script for live phishing detection" query="Draft a Streamlit app.py script for this phishing email detection project" query_intent="CLICKABLE_SUGGESTION" />
<Elicitation label="Generate unit tests for text preprocessing functions" query="Generate pytest unit tests for the preprocess_email_text function" query_intent="CLICKABLE_SUGGESTION" />
<Elicitation label="Create a requirements.txt file with exact dependencies" query="Create a requirements.txt file for the phishing email detection project" query_intent="CLICKABLE_SUGGESTION" />
</ElicitationsGroup>
