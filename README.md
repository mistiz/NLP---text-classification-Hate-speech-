Twitter Hate Speech Classification
A Complete Academic Guide for Beginners in NLP
Natural Language Processing | Phase I: Classical Machine Learning

 
1. Introduction — What Is This Project?
This document is a step-by-step academic guide to building a hate speech text classifier using Natural Language Processing (NLP). It is written for someone new to NLP who wants to understand not just what the code does, but the reasoning behind every decision — why certain techniques were chosen, what alternatives exist, and where the field is heading.

The project classifies tweets from Twitter as either hate speech (label = 1) or non-hate speech (label = 0). The dataset contains 1,614 labeled tweets. This is a binary text classification problem — one of the most fundamental tasks in NLP.

1.1 What Is Hate Speech?
Hate speech is defined by three simultaneous conditions that must all be true:
1.	It is a deliberate attack
2.	It is directed towards a specific group of people
3.	It is motivated by aspects of the group's identity — such as race, religion, ethnicity, or origin

A critical challenge: the word 'hate' appears in both hate speech and non-hate speech tweets. 'I hate this weather' is not hate speech. This demonstrates why simple keyword matching fails and why more sophisticated methods are required.

 
2. The Four Paradigms of NLP
NLP has evolved through four major paradigms, each building on the limitations of the previous. Understanding these paradigms is essential for knowing where any given technique fits.

Paradigm 1 — Rule-Based NLP (Pre-2000s)
Hand-crafted rules written by linguists. No machine learning. Works by pattern matching — if a tweet contains a slur from a predefined list, flag it.
•	Strength: Fully transparent, requires no training data
•	Weakness: Cannot understand context — 'I hate this weather' gets wrongly flagged
•	Accuracy on this task: approximately 55–60%

Paradigm 2 — Classical Machine Learning (2000s–2014)
Learn patterns from labeled data using hand-crafted features. You engineer the features (TF-IDF, n-grams, sentiment scores), then train a statistical model (Logistic Regression, SVM, Naive Bayes). This is exactly what Phase I of this project implements.
•	Strength: Works well on small datasets, fast to train, interpretable
•	Weakness: You decide what features matter — the model cannot discover new ones
•	Accuracy on this task: approximately 65–75%

Paradigm 3 — Deep Learning / Neural NLP (2014–2018)
Let the model learn its own features using neural networks. Word embeddings (Word2Vec, GloVe, FastText) feed into neural networks such as CNNs and LSTMs. The model learns what matters automatically.
•	Strength: Learns complex patterns automatically, handles word order and sequence
•	Weakness: Needs substantially more labeled data than classical methods
•	Accuracy on this task: approximately 65–79% depending on architecture

Paradigm 4 — Pretrained Transformers / LLMs (2018–present)
Pretrain a massive model on billions of words, then fine-tune or prompt it for a specific task. Models like BERT and GPT understand every word in the context of every other word simultaneously using self-attention.
•	Strength: Understands context deeply, state of the art on almost every NLP task
•	Weakness: Computationally expensive, requires GPU for fine-tuning
•	Accuracy on this task: approximately 82–95% depending on model

 
3. All Approaches to Hate Speech Classification
The table below covers every major approach from oldest to most modern, organized by paradigm. Rows 3, 4, and 5 show your actual Phase I results.

#	Approach	Paradigm	Technology	Core Math	Accuracy	Strengths	Weaknesses
1	Keyword Rule-Based	1 — Rule-Based	Regex, slur lists	If-else, pattern matching	55–60%	No data needed, transparent	Cannot understand context
2	Bag of Words + Naive Bayes	2 — Classical ML	CountVectorizer, MultinomialNB	Bayes theorem	65–68%	Fast, works on small data	Assumes word independence
3	TF-IDF + Logistic Regression	2 — Classical ML	TfidfVectorizer, sklearn	Log-odds, L2 regularization	72.81% (your result)	Interpretable, tunable, strong baseline	No semantic understanding
4	TF-IDF + Linear SVM	2 — Classical ML	TfidfVectorizer, LinearSVC	Max margin hyperplane, hinge loss	71.25% (your result)	Strong on sparse text	Lower non-hate recall
5	TF-IDF + Naive Bayes	2 — Classical ML	TfidfVectorizer, MultinomialNB	Bayes theorem, conditional probability	66.56% (your result)	Very fast, computationally efficient	Critical recall imbalance: 0.44 non-hate
6	TF-IDF + Random Forest	2 — Classical ML	RandomForestClassifier	Ensemble of trees, Gini impurity	68–72%	Handles non-linear patterns	Slow on large feature sets
7	TF-IDF + XGBoost	2 — Classical ML	XGBClassifier	Boosted trees, cross-entropy	70–74%	Strong on tabular data	Many hyperparameters
8	Word2Vec + LR	3 — Neural Embeddings	Gensim Word2Vec	Skip-gram, average pooling	68–72%	Captures semantic similarity	Averages lose word order
9	FastText + LR	3 — Neural Embeddings	Facebook FastText	Subword n-grams, softmax	70–74%	Handles misspellings, great for Twitter	Static, no deep context
10	DNN on TF-IDF	3 — Deep Learning	TensorFlow/Keras	Dense layers, dropout, sigmoid	70–75%	Easy transition from sklearn	No sequence understanding
11	CNN on Embeddings	3 — Deep Learning	TensorFlow/Keras	Conv filters, max pooling	73–77%	Captures local n-gram patterns	Needs more data
12	BiLSTM + Attention	3 — Deep Learning	TensorFlow/Keras	Bidirectional LSTM, attention weights	75–79%	Handles word order and sequence	Slow to train
13	BERT Fine-tuned	4 — Transformers	HuggingFace, bert-base-uncased	Self-attention, masked LM	82–87%	Deep context, handles sarcasm and negation	Needs GPU
14	HateBERT Fine-tuned	4 — Transformers	HuggingFace, GroNLP/hateBERT	BERT pretrained on abusive content	85–90%	Domain-specific for hate speech	Narrow domain
15	Zero-shot LLM	4 — Transformers	OpenAI / Anthropic API	In-context learning	75–82%	No training needed	Expensive, non-deterministic
16	Few-shot LLM	4 — Transformers	OpenAI / Anthropic API	In-context learning + examples	78–84%	Better than zero-shot	Prompt sensitive
17	Hybrid TF-IDF + VADER + LR	2+3 — Hybrid	TfidfVectorizer, VADER, scipy hstack	Feature concatenation, LR	74–76% (your Phase I+h)	Combines lexical and sentiment	No deep context
18	Embedding + Classifier	5 — Modern LLM	Sentence-BERT, OpenAI embeddings	Dense embeddings + small classifier	85–93%	Very strong, tiny dataset friendly	Requires embedding API
19	LLM Fine-tuning (LoRA)	5 — Modern LLM	HuggingFace PEFT, LoRA/QLoRA	Low-rank adaptation	90–95%	Best performance, customizable	Requires GPU

 
4. Generation Summary — Oldest to Latest
A simplified comparison of each generation focusing on key properties that determine when each approach is appropriate.

Method	Generation	Uses TF-IDF?	Learns Context?	Data Needed	Typical Accuracy
Logistic Regression	1 — Classical ML	Yes	No	Very small	70–75%
Linear SVM	1 — Classical ML	Yes	No	Very small	70–74%
Naive Bayes	1 — Classical ML	Yes	No	Very small	60–70%
DNN on TF-IDF	3 — Deep Learning	Yes	No	Small	70–75%
LSTM	3 — Deep Learning	No	Partial	Medium	65–72%
CNN	3 — Deep Learning	No	Partial	Medium	68–75%
BERT Fine-tuned	4 — Transformers	No	Yes (deep)	Small	85–92%
DistilBERT	4 — Transformers	No	Yes (deep)	Small	83–90%
LLM Zero-shot	5 — Modern LLM	No	Yes (deep)	None	80–90%
Embedding + Classifier	5 — Modern LLM	No	Yes (deep)	Very small	85–93%
LLM Fine-tuning (LoRA)	5 — Modern LLM	No	Yes (deep)	Small	90–95%

 
5. The Project Pipeline — Step by Step
This section walks through the exact pipeline used in Phase I, explaining what each step does and why it was done in this order.

Step 1 — Data Loading and Sanity Checks
The dataset (train.csv) contains two columns: text (the raw tweet) and label (0 or 1). Before any analysis, basic sanity checks confirm: shape of the dataset, presence of missing values, and that labels are binary integers. The dataset has 1,614 rows with zero missing values.
df = pd.read_csv('train.csv')
print(df.shape)           # (1614, 2)
print(df.isnull().sum())  # text: 0, label: 0

Step 2 — Data Exploration (Task 1a)
Exploration goes beyond basic statistics. Five specific analyses were performed:
4.	Class balance analysis — 50.56% hate / 49.44% non-hate. Nearly perfectly balanced. Accuracy is a fair metric. No oversampling needed.
5.	Text length analysis — hate speech tweets average 23.6 words vs 17.0 words for non-hate. Hate speech tends to be more elaborate.
6.	Top-K words per class — common words like 'people' appear in both classes, confirming raw frequency is insufficient.
7.	Exclusive words analysis — 2,117 words appear only in hate speech and never in non-hate speech, identifying the most discriminative vocabulary.
8.	Sample tweets from each class — qualitative inspection to understand the nature of the content.

Step 3 — Data Preprocessing (Task 1b)
A ten-step preprocessing pipeline applied consistently to both training and inference data:
9.	Lowercasing — standardizes vocabulary so 'Hate' and 'hate' are the same word
10.	URL removal — links carry no hate-speech signal
11.	@mention removal — identity of mentioned users is irrelevant to the label
12.	Hashtag handling — remove # symbol but keep the word
13.	Emoji removal — sparse in this small dataset, adds noise for bag-of-words models
14.	Punctuation removal — reduces noise for TF-IDF features
15.	Digit removal — numbers rarely carry meaningful hate speech signal
16.	Whitespace normalization — removes extra spaces
17.	Stopword removal — common words like 'the', 'a', 'in' appear equally in both classes
18.	Word-length filtering — removes tokens shorter than 3 characters

Lemmatization using spaCy was also applied after the pipeline, normalizing words to their base form. After preprocessing, 17 rows were dropped as empty, leaving 1,597 usable rows.

Step 4 — Train/Test Split
The dataset was split into 80% training (1,277 rows) and 20% testing (320 rows) using stratified splitting. Stratification ensures the 50/50 class ratio is preserved in both splits.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

CRITICAL: The split must happen BEFORE fitting the vectorizer. Fitting TF-IDF on the full dataset before splitting is data leakage — the model gains unfair knowledge of the test set vocabulary.

Step 5 — TF-IDF Vectorization
Text is converted to numerical vectors. TF-IDF penalizes words that appear in all documents (like 'the') and rewards words distinctive to specific documents (like 'parasite'). The vectorizer is fit ONLY on training data.
tfidf = TfidfVectorizer(
    max_features=10000,   # top 10,000 terms
    ngram_range=(1, 2),   # unigrams + bigrams
    stop_words='english', # remove common words
    sublinear_tf=True     # apply log(tf)
)
X_train_vec = tfidf.fit_transform(X_train)  # fit on train ONLY
X_test_vec  = tfidf.transform(X_test)       # transform test only

Step 6 — Logistic Regression with L2 Regularization
With 10,000 features and only 1,277 training rows, overfitting is a risk. L2 regularization adds a penalty for large weights. A grid search over six lambda values was performed. In scikit-learn, C = 1/lambda.
lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

Results: lambda = 0.0001 was optimal with 72.81% accuracy. This was hard-coded as required by the assignment.

Step 7 — Three Model Comparison
Three models were evaluated against the 65% accuracy target:
•	Logistic Regression (lambda = 0.0001): 72.81% — BEST, most balanced precision and recall
•	Linear SVM: 71.25% — strong but lower non-hate recall (0.62)
•	Multinomial Naive Bayes: 66.56% — exceeds target but critically low non-hate recall (0.44)

Logistic Regression was selected as the final model. Naive Bayes, despite exceeding the accuracy threshold, showed unacceptable recall imbalance — flagging the majority of innocent tweets as hate speech. In real content moderation, this would wrongly remove legitimate content at scale.

 
6. Optional Advancement — Feature Engineering
Beyond the three required techniques (stopword removal, bigrams, TF-IDF), thirteen additional features were explored. The table below summarizes all optional advancement work.

Feature (Category)	What It Does	Implemented?	Why It Helps
Stopword removal (e)	Removes 'the, a, in' during TF-IDF	Yes	Reduces noise; improves model focus on meaningful words
Unigrams + bigrams (f)	Captures multi-word expressions	Yes (bigrams)	Bigrams capture phrases like 'go back', 'hate group'
TF-IDF weighting (g)	Weighted features instead of raw counts	Yes	Highlights rare, discriminative vocabulary
Character n-grams (h)	Captures misspellings, obfuscated slurs	Yes	Strong for noisy Twitter text
VADER sentiment scores (h)	Adds polarity score per tweet	Yes	Detects aggressive tone beyond vocabulary
Emoji counting (h)	Counts emojis as emotional signal	Yes	Emojis signal sarcasm, anger, or hostility
Lemmatization — spaCy (h)	Normalizes words to base form	Yes	Reduces vocabulary sparsity; improves generalization
Word2Vec embeddings (h)	Dense semantic vectors from dataset	Yes	Captures semantic similarity beyond TF-IDF
FastText embeddings (h)	Subword-aware embeddings	Yes	Handles misspellings common in Twitter text
POS tag features (h)	Noun, verb, adjective counts per tweet	Yes	Captures syntactic patterns of hate speech
Combined feature matrix (h)	TF-IDF + VADER merged via hstack	Yes	Richer representation combining two signal types
Error analysis (h)	Analyzes misclassified tweets	Yes	Identifies model weaknesses and failure patterns
Word-length filtering (h)	Removes tokens under 3 characters	Yes	Eliminates uninformative fragments beyond stopwords
Class weighting / oversampling (h)	Handles class imbalance	Not needed	Dataset is already balanced at 50.56% / 49.44%

 
7. Why We Made These Decisions
Every technical decision was driven by specific properties of the dataset and task.

Project Property	What It Told Us	Decision Made	What Was Ruled Out
Dataset size: 1,614 rows	Too small for deep learning	Use classical ML — LR and LinearSVC	CNNs, LSTMs — need tens of thousands of rows
Balanced classes (50/50)	Accuracy is a fair metric	Use accuracy as primary metric	SMOTE, oversampling, class weighting
Short tweets — avg 15–20 words	Very little context per sample	Use bigrams to capture short phrases	Trigrams — exponential cost, minimal gain
Noisy Twitter text — misspellings	Word-level models miss obfuscated words	Add character n-grams and FastText	Pure word-level bag of words alone
Context-dependent hate speech	'hate' appears in both classes	Use TF-IDF and bigrams for context	Simple keyword matching
10,000 features, 1,277 rows	Overfitting risk	L2 regularization — best lambda = 0.0001	No regularization — memorizes training data
Emotional tone in tweets	Vocabulary alone misses aggression	Add VADER sentiment scores	Using only lexical features
Assignment: classical models only	No transformers in Phase I	Stay in Paradigm 2 and 3	BERT, RoBERTa — deferred to Phase II
Small training set: 1,277 rows	SVM needs more support vectors	LR wins over SVM on this size	Relying solely on SVM
Colab environment	Large downloads unreliable	Remove GloVe (862MB unstable)	GloVe — replaced by Word2Vec and FastText
Deliberate misspellings in hate speech	Word models miss 'h8', 'stuuupid'	Character n-grams (3–5) + FastText	Word-level only approaches

 
8. TensorFlow Alternative Implementation
The same classifier can be built in TensorFlow by replacing Logistic Regression with a dense neural network. The pipeline maps almost identically.

8.1 When to Use TensorFlow vs scikit-learn
•	Use scikit-learn when: dataset is small, interpretability matters, fast iteration is needed
•	Use TensorFlow when: experimenting with neural architectures, adding embedding layers, or scaling to larger datasets

8.2 TensorFlow Code
Step 1 — Convert sparse TF-IDF matrix to dense tensor:
X_train_tf = tf.convert_to_tensor(X_train_vec.toarray(), dtype=tf.float32)
X_test_tf  = tf.convert_to_tensor(X_test_vec.toarray(), dtype=tf.float32)

Step 2 — Build a simple dense neural network:
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_tf, y_train_tf, validation_split=0.1, epochs=10, batch_size=32)

8.3 Expected Performance
•	Logistic Regression (scikit-learn): 72.81% (actual result)
•	TensorFlow Dense Neural Network: ~70–75%
•	LSTM / CNN on this small dataset: ~60–68% (too little data)
•	BERT fine-tuned (Phase II): ~85–92%

 
9. Approach, Decisions, and Future Directions
Why We Started With Classical ML
The dataset of 1,614 tweets was small, balanced, and noisy. Classical ML was the natural starting point: fast to train, interpretable, and proven on small datasets. Deep learning typically requires tens of thousands of labeled examples to outperform classical methods reliably — this dataset did not meet that threshold.
TF-IDF was chosen because raw word counts treat all words equally. 'The' and 'parasite' would carry the same weight despite one being meaningless and the other highly discriminative. TF-IDF corrects this. Bigrams extended the approach by capturing phrases like 'go back' or 'hate group' that carry stronger signal than either word alone. Lambda = 0.0001 was found optimal through grid search, indicating the model benefited from very weak regularization on this clean, balanced dataset.

What Was Not Done and Why
Naive Bayes was implemented and evaluated but showed a critical imbalance — non-hate recall of 0.44 makes it unsuitable for real-world content moderation despite exceeding the accuracy threshold. GloVe embeddings were removed due to the 862MB download being unreliable in Colab. Trigrams were not added because they increase computational complexity exponentially for marginal gain on short tweets. Deep learning architectures were deferred because the dataset was too small. Transformer models were deferred to Phase II as required by the assignment.

What Could Be Done Next
Phase II applies a large language model to the same task. HateBERT, pretrained on abusive Reddit content, would be the strongest choice for fine-tuning and would likely achieve 85–90% accuracy compared to the 72.81% achieved in Phase I. Zero-shot prompting with GPT-4 or Claude tests how well general language understanding transfers without any training. Beyond Phase II, explainability tools like LIME or SHAP would make model decisions transparent — essential for responsible deployment in content moderation.

 
10. Final Results

Model	Lambda	Accuracy	Non-Hate Precision	Non-Hate Recall	Hate Precision	Hate Recall
Logistic Regression	0.0001	72.81% (BEST)	0.74	0.67	0.72	0.78
Linear SVM	N/A	71.25%	0.75	0.62	0.69	0.80
Multinomial Naive Bayes	N/A	66.56%	0.78	0.44 (critical)	0.62	0.88

All three models exceeded the 65% accuracy target. Logistic Regression with lambda = 0.0001 was selected as the final model. The critical insight from this comparison is that accuracy alone is insufficient for evaluating content moderation models. Naive Bayes achieved 66.56% accuracy but its non-hate recall of 0.44 means it would wrongly flag over half of all innocent content — an unacceptable outcome in any real deployment.

 
11. Key Concepts Explained Simply
What is TF-IDF?
TF-IDF assigns a weight to each word based on two things: how often it appears in this specific tweet (TF), and how rare it is across all tweets (IDF). Words that appear everywhere like 'the' get a low score. Words distinctive to specific content like 'parasite' get a high score. This makes TF-IDF far more informative than raw word counts.

What is Data Leakage?
Data leakage occurs when information from the test set influences training. The most common form in NLP is fitting a vectorizer on the full dataset before splitting. The correct approach: fit the vectorizer only on training data, then transform both training and test data separately.

What is L2 Regularization?
With 10,000 features and 1,277 training rows, the model risks overfitting — memorizing training examples rather than learning general patterns. L2 regularization adds a penalty for large weights, forcing the model to stay simple and generalize better. In scikit-learn, C = 1/lambda. Smaller C means stronger regularization. The optimal lambda of 0.0001 (very weak regularization) indicates the features were informative enough that strong penalization was not needed.

Why Does Recall Matter More Than Accuracy?
Accuracy tells you what percentage of all predictions were correct. Recall tells you what percentage of actual positives were correctly identified. In content moderation, a model with 66% accuracy but 0.44 non-hate recall would wrongly flag 56% of innocent tweets. A model with 72% accuracy and 0.67 non-hate recall is far safer in practice. This is why Logistic Regression was chosen over Naive Bayes despite both exceeding the accuracy target.

 
12. What Comes Next — Phase II and III
Phase II — LLM-Based Approach
Phase II applies a large language model to the same task using this pipeline:
19.	Data ingestion — same train.csv
20.	Light preprocessing — LLMs understand raw Twitter text without heavy cleaning
21.	Task formulation — define the classification task for the model
22.	Prompt design — craft the instruction given to the model
23.	Prompt testing and iteration — test on 10–20 examples before full inference
24.	Model selection — HateBERT, DistilBERT, GPT-4, or Claude
25.	Batch inference — process all tweets through the model
26.	Post-processing — convert text output to clean 0/1 labels
27.	Evaluation — same metrics as Phase I for direct comparison
28.	Error analysis — examine misclassified tweets
29.	Hybrid or fine-tuned approach — combine Phase I and Phase II models

Phase III — Demo Application
A user-facing application where anyone can type a tweet and receive a real-time prediction. The inference function must apply the same preprocessing pipeline used during training, guaranteeing consistency between training and deployment.

The Bigger Picture
This project traces the full arc of modern NLP. Phase I shows that classical methods, properly engineered, produce strong results on small datasets. The gap between Logistic Regression (72.81%) and BERT (85–90%) represents the cost of not having contextual understanding — a cost that becomes visible when dealing with sarcasm, coded language, and ambiguous phrasing. Phase II closes that gap. Phase III makes it useful.

CSC 7650 — NLP Project Phase I  |  Twitter Hate Speech Classification  |  Academic Reference Guide  |  Final Results: LR 72.81%, SVM 71.25%, NB 66.56%
