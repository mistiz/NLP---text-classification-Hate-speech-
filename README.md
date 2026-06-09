# Twitter Hate Speech Text Classification

> A complete NLP pipeline for detecting hate speech in tweets — from classical machine learning to LLM-based approaches.

---

## Project Overview

This project builds a binary text classification system that detects hate speech in Twitter data. A tweet is classified as hate speech (label = 1) or non-hate speech (label = 0) based on three simultaneous conditions:

1. It is a deliberate attack
2. It is directed towards a specific group of people
3. It is motivated by aspects of the group's identity — such as race, religion, ethnicity, or origin

**Dataset:** 1,614 labeled tweets — 50.56% hate speech, 49.44% non-hate speech (nearly perfectly balanced)

---

## Project Structure

```
├── Hate_Speech_Text_Classification.ipynb   # Main notebook — Phase I
├── train.csv                               # Labeled tweet dataset
└── README.md                               # This file
```

---

## Phase I — Classical Machine Learning

### Pipeline

```
Raw Data (train.csv)
        ↓
Load Data + Sanity Checks
        ↓
Data Exploration
        ↓
Text Preprocessing (10 steps + lemmatization)
        ↓
Train/Test Split (80/20, stratified)
        ↓
TF-IDF Vectorization (fit on train only)
        ↓
Optional Feature Engineering (VADER, FastText, Word2Vec, POS tags, char n-grams)
        ↓
Logistic Regression + L2 Grid Search
        ↓
Linear SVM
        ↓
Multinomial Naive Bayes
        ↓
Three-Model Comparison
        ↓
Error Analysis
        ↓
Inference Function
        ↓
Final Model: Logistic Regression (λ=0.0001) — 72.81% Accuracy
```

---

## Final Results

| Model | Accuracy | Non-Hate Recall | Hate Recall |
|---|---|---|---|
| **Logistic Regression (λ=0.0001)** | **72.81% ✅** | 0.67 | 0.78 |
| Linear SVM | 71.25% | 0.62 | 0.80 |
| Multinomial Naive Bayes | 66.56% | 0.44 ⚠️ | 0.88 |

All three models exceed the 65% accuracy target. Logistic Regression was selected as the final model due to its highest accuracy and most balanced recall across both classes.

> **Why recall matters more than accuracy:** Naive Bayes achieved 66.56% accuracy but its non-hate recall of 0.44 means it would wrongly flag over half of all innocent tweets as hate speech — unacceptable in any real deployment.

---

## Preprocessing Pipeline

Each tweet was cleaned using a consistent ten-step pipeline applied to both training and inference:

| Step | Operation | Reason |
|---|---|---|
| 1 | Lowercasing | Standardizes vocabulary |
| 2 | URL removal | Links carry no hate-speech signal |
| 3 | @mention removal | Identity of mentioned users is irrelevant |
| 4 | Hashtag handling | Remove # but keep word |
| 5 | Emoji removal | Sparse in this dataset, adds noise |
| 6 | Punctuation removal | Reduces noise for bag-of-words models |
| 7 | Digit removal | Numbers rarely carry hate speech signal |
| 8 | Whitespace normalization | Removes extra spaces |
| 9 | Stopword removal | Common words appear equally in both classes |
| 10 | Word-length filtering | Removes uninformative short fragments |

Lemmatization using spaCy was applied after the pipeline, normalizing words to their base form.

---

## Feature Engineering

### Required (e, f, g)

| Feature | Implementation |
|---|---|
| Stopword removal | `stop_words='english'` in TfidfVectorizer |
| Unigrams + bigrams | `ngram_range=(1,2)` in TfidfVectorizer |
| TF-IDF weighting | TfidfVectorizer with `sublinear_tf=True` |

### Optional Advancement (h)

| Feature | What It Does |
|---|---|
| Character n-grams | Captures misspellings and obfuscated slurs |
| VADER sentiment scores | Adds polarity score to capture aggressive tone |
| Emoji counting | Counts emojis as proxy for emotional intensity |
| Lemmatization | Normalizes words to base form |
| POS tag features | Noun, verb, adjective counts per tweet |
| Word2Vec embeddings | Dense semantic vectors trained on dataset |
| FastText embeddings | Subword-aware embeddings for noisy Twitter text |
| Word-length filtering | Removes tokens under 3 characters |
| Error analysis | Analyzes misclassified tweets for patterns |
| Combined feature matrix | TF-IDF + VADER merged via scipy hstack |

---

## Setup and Installation

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk spacy gensim
pip install vaderSentiment emoji fasttext-wheel
python -m spacy download en_core_web_sm
```

### Running in Google Colab

1. Upload `train.csv` via the Colab sidebar (folder icon → upload)
2. Open `Hate_Speech_Text_Classification.ipynb`
3. Run **Runtime → Restart and Run All**

---

## Key Concepts

**TF-IDF** — Assigns higher weight to words that are distinctive to specific tweets and lower weight to words that appear everywhere. Makes "parasite" more important than "the".

**Data Leakage** — Fitting the vectorizer on the full dataset before splitting gives the model unfair knowledge of the test set. The vectorizer must be fit on training data only.

**L2 Regularization** — With 10,000 features and 1,277 training rows, overfitting is a risk. L2 regularization penalizes large weights and forces the model to generalize. In sklearn: `C = 1/lambda`.

**Stratified Split** — Random splitting might produce an imbalanced test set by chance. Stratification guarantees both splits preserve the original class ratio.

---

## The Four Paradigms of NLP

| Paradigm | Era | Example | Accuracy on This Task |
|---|---|---|---|
| Rule-Based | Pre-2000s | Regex, slur lists | ~55–60% |
| Classical ML | 2000s–2014 | TF-IDF + Logistic Regression | ~65–75% |
| Deep Learning | 2014–2018 | Word2Vec + LSTM | ~65–79% |
| Transformers | 2018–present | BERT, GPT, Claude | ~82–95% |

This project implements Paradigm 2 (Phase I) with optional Paradigm 3 features, and will extend to Paradigm 4 in Phase II.

---

## All Approaches Compared

| Approach | Paradigm | Accuracy | Context-Aware? |
|---|---|---|---|
| Keyword Rule-Based | 1 | 55–60% | No |
| TF-IDF + Naive Bayes | 2 | 65–68% | No |
| TF-IDF + Logistic Regression | 2 | **72.81%** | No |
| TF-IDF + Linear SVM | 2 | 71.25% | No |
| Word2Vec + LR | 3 | 68–72% | Partial |
| FastText + LR | 3 | 70–74% | Partial |
| BERT Fine-tuned | 4 | 82–87% | Yes |
| HateBERT Fine-tuned | 4 | 85–90% | Yes |
| LLM Zero-shot | 4 | 75–82% | Yes |
| LLM Fine-tuning (LoRA) | 5 | 90–95% | Yes |

---

## What Comes Next

**Phase II — LLM-Based Approach**

Apply a large language model (Gemini / HateBERT) to the same task using:
- Zero-shot prompting (baseline)
- Few-shot in-context learning
- Structured Chain-of-Thought reasoning based on the three-part hate speech definition
- Direct comparison with Phase I results

**Phase III — Demo Application**

A user-facing interface where anyone can type a tweet and receive a real-time prediction with confidence score.

---

## Ethical Note

This project involves sensitive content and is built for academic purposes only. The dataset contains real examples of hate speech and should be handled responsibly. Model predictions should not be used as the sole basis for content moderation decisions without human review.

---

## Course

 Natural Language Processing | Phase I: Classical Machine Learning
