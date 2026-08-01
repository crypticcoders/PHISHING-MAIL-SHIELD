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
